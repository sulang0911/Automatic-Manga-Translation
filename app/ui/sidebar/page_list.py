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
from app.ui.widgets.thumbnail_loader import AsyncThumbnailManager
from app.ui.theme.icons import get_icon
from app.core.cache.cache_manager import get_cache_manager

_RE_DIGITS = re.compile(r'(\d+)')


def natural_sort_key(s: str) -> list:
    """Natural alphanumeric sort key, sorting 'page2.png' before 'page10.png'."""
    return [int(c) if c.isdigit() else c.lower() for c in _RE_DIGITS.split(os.path.basename(s))]


def natural_sort_path_key(s: str) -> list:
    """
    Natural alphanumeric sort key preserving directory hierarchy and sorting
    both directory names and file names naturally (e.g. vol1/02.png before vol1/10.png,
    and vol1/10.png before vol2/01.png).
    """
    norm = os.path.normpath(s).replace("/", os.sep).replace("\\", os.sep)
    parts = norm.split(os.sep)
    return [[int(c) if c.isdigit() else c.lower() for c in _RE_DIGITS.split(part)] for part in parts]


def is_ignored_cache_or_export(file_path: str, base_dir: Optional[str] = None) -> bool:
    """
    Returns True if the path or file is an internal cache folder, hidden file/directory,
    or intermediate cached file (.erased.*, .rendered.*, _erased.*, _translated.*, .blocks.json).
    """
    norm = os.path.normpath(os.path.abspath(file_path))
    known_cache_dirs = {"translation_cache", "__pycache__", ".amt_cache", ".git", ".cache"}

    if base_dir:
        abs_base = os.path.normpath(os.path.abspath(base_dir))
        try:
            rel = os.path.relpath(norm, abs_base)
            rel_parts = rel.split(os.sep)
        except ValueError:
            rel_parts = norm.split(os.sep)
        for part in rel_parts[:-1]:
            part_lower = part.lower()
            if part.startswith(".") or part_lower in known_cache_dirs:
                return True
        for part in abs_base.split(os.sep):
            if part.lower() in known_cache_dirs:
                return True
    else:
        parts = norm.split(os.sep)
        for part in parts[:-1]:
            if part.lower() in known_cache_dirs:
                return True
        if len(parts) >= 2 and parts[-2].startswith("."):
            return True

    fname = os.path.basename(norm).lower()
    if fname.startswith("."):
        return True

    if fname.endswith(".blocks.json"):
        return True

    # Intermediate cached files: .erased.*, .rendered.*, _erased.*, _translated.*
    if ".erased." in fname or fname.endswith(".erased"):
        return True
    if ".rendered." in fname or fname.endswith(".rendered"):
        return True
    if "_erased." in fname or fname.endswith("_erased"):
        return True
    if "_translated." in fname or fname.endswith("_translated"):
        return True

    return False


class PageItemWidget(QWidget):
    """Custom item renderer for each manga page in the queue."""
    sig_remove = pyqtSignal(str)
    sig_edit_style = pyqtSignal(dict)

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

        # Thumbnail (Click to open page-level style dialog)
        self.thumb_label = QLabel(self)
        self.thumb_label.setObjectName("thumbLabel")
        self.thumb_label.setFixedSize(36, 48)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.thumb_label.setToolTip("点击修改此页文字排版与样式配置")
        self.thumb_label.setStyleSheet(
            "background-color: rgba(255, 255, 255, 0.04);"
            "border-radius: 4px;"
            "border: 1px solid rgba(255, 255, 255, 0.08);"
            "color: #71717A;"
            "font-size: 13px;"
        )
        self.thumb_label.setText("📄")
        self._load_thumbnail(item_data["path"])
        layout.addWidget(self.thumb_label)

        # Text column: filename and status caption
        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)

        self.name_label = QLabel(os.path.basename(item_data["path"]), self)
        self.name_label.setStyleSheet("font-weight: 500; font-size: 12px;")
        if item_data.get("rel_path"):
            self.setToolTip(item_data["rel_path"])
            self.name_label.setToolTip(item_data["rel_path"])
        text_col.addWidget(self.name_label)

        self.status_label = QLabel(item_data.get("status_text", "等待中"), self)
        self.status_label.setStyleSheet("font-size: 11px; opacity: 0.65;")
        text_col.addWidget(self.status_label)

        layout.addLayout(text_col, 1)

        # Style button (palette / sparkles icon)
        self.style_btn = QToolButton(self)
        self.style_btn.setIcon(get_icon("sparkles", color="#3B82F6", size=14))
        self.style_btn.setToolTip("修改此页文字设置 (排版/字体/粗体)")
        self.style_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.style_btn.setStyleSheet("border: none; background: transparent; padding: 2px;")
        self.style_btn.clicked.connect(lambda: self.sig_edit_style.emit(self.item_data))
        layout.addWidget(self.style_btn)

        # Remove button
        self.remove_btn = QToolButton(self)
        self.remove_btn.setIcon(get_icon("trash", color="#71717A", size=14))
        self.remove_btn.setToolTip("从队列中移除")
        self.remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.remove_btn.setStyleSheet("border: none; background: transparent; padding: 2px;")
        self.remove_btn.clicked.connect(lambda: self.sig_remove.emit(self.item_id))
        layout.addWidget(self.remove_btn)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            child = self.childAt(event.pos())
            if child == self.thumb_label:
                self.sig_edit_style.emit(self.item_data)

    def _load_thumbnail(self, path: str):
        if not path:
            return

        def _on_ready(pix: QPixmap):
            try:
                if not pix.isNull():
                    self.thumb_label.setText("")
                    self.thumb_label.setStyleSheet(
                        "background-color: transparent;"
                        "border-radius: 4px;"
                        "border: 1px solid rgba(255, 255, 255, 0.08);"
                    )
                    self.thumb_label.setPixmap(pix)
            except RuntimeError:
                pass

        AsyncThumbnailManager.instance().request_thumbnail(path, (36, 48), _on_ready)

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
    sig_edit_page_style = pyqtSignal(dict)
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
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
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
        across subdirectories and deduplicating against existing entries.
        Preserves relative subfolder hierarchy and root directory metadata.
        """
        valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        discovered_entries: List[Dict[str, str]] = []

        valid_input_paths = [p for p in paths if p and os.path.exists(p)]
        dir_paths = [os.path.normpath(os.path.abspath(p)) for p in valid_input_paths if os.path.isdir(p)]

        batch_root: Optional[str] = None
        if len(dir_paths) > 1:
            try:
                common = os.path.commonpath(dir_paths)
                if os.path.isdir(common) and all(p != common for p in dir_paths):
                    batch_root = common
            except Exception:
                batch_root = None

        for p in valid_input_paths:
            p_abs = os.path.normpath(os.path.abspath(p))

            if os.path.isfile(p_abs):
                if is_ignored_cache_or_export(p_abs):
                    continue
                ext = os.path.splitext(p_abs)[1].lower()
                if ext in valid_extensions:
                    discovered_entries.append({
                        "path": p_abs,
                        "rel_path": os.path.basename(p_abs),
                        "root_dir": os.path.dirname(p_abs),
                    })
            elif os.path.isdir(p_abs):
                effective_root = batch_root if batch_root else p_abs
                dir_name = os.path.basename(p_abs).lower()
                if dir_name.startswith(".") or dir_name in ("translation_cache", "__pycache__", ".amt_cache"):
                    continue

                for root, dirs, files in os.walk(p_abs):
                    # Prune hidden directories and internal cache directories in-place
                    dirs[:] = [
                        d for d in dirs
                        if not d.startswith(".") and d.lower() not in ("translation_cache", "__pycache__", ".amt_cache")
                    ]
                    dirs.sort(key=lambda d: [int(c) if c.isdigit() else c.lower() for c in _RE_DIGITS.split(d)])

                    for f in files:
                        full_f_path = os.path.normpath(os.path.join(root, f))
                        if is_ignored_cache_or_export(full_f_path, base_dir=effective_root):
                            continue
                        ext = os.path.splitext(f)[1].lower()
                        if ext in valid_extensions:
                            rel_p = os.path.relpath(full_f_path, effective_root)
                            discovered_entries.append({
                                "path": full_f_path,
                                "rel_path": rel_p,
                                "root_dir": effective_root,
                            })

        # Deduplicate while preserving order of discovery
        seen_paths = set()
        unique_entries = []
        for entry in discovered_entries:
            p = entry["path"]
            if p not in seen_paths:
                seen_paths.add(p)
                unique_entries.append(entry)

        # Natural alphanumeric sorting across subdirectories
        unique_entries.sort(key=lambda it: natural_sort_path_key(it["path"]))

        existing_paths = {item["path"] for item in self.items_data}
        new_items = []
        cache_mgr = get_cache_manager()
        for entry in unique_entries:
            path = entry["path"]
            if path not in existing_paths:
                existing_paths.add(path)
                c_status = cache_mgr.has_cache(path)
                if (c_status["blocks"] and (c_status["rendered"] or c_status["erased"])) and cache_mgr.is_fully_translated(path):
                    status = "completed"
                    status_text = "已完成(缓存)"
                elif c_status["erased"]:
                    status = "processing"
                    status_text = "已擦除(缓存)"
                else:
                    status = "queued"
                    status_text = "等待中"

                item_data = {
                    "id": str(uuid.uuid4()),
                    "path": path,
                    "rel_path": entry["rel_path"],
                    "root_dir": entry["root_dir"],
                    "status": status,
                    "status_text": status_text,
                }
                new_items.append(item_data)
                self.items_data.append(item_data)

        # Populate GUI items with bulk updates suspended for maximum performance
        self.list_widget.setUpdatesEnabled(False)
        try:
            for data in new_items:
                list_item = QListWidgetItem(self.list_widget)
                list_item.setSizeHint(QSize(200, 56))
                widget = PageItemWidget(data, self.list_widget)
                widget.sig_remove.connect(self.remove_item)
                widget.sig_edit_style.connect(self.sig_edit_page_style.emit)
                self.list_widget.addItem(list_item)
                self.list_widget.setItemWidget(list_item, widget)
                self._item_widgets[data["id"]] = widget
        finally:
            self.list_widget.setUpdatesEnabled(True)

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
        AsyncThumbnailManager.instance().clear_cache()
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

    def _on_item_double_clicked(self, item: QListWidgetItem):
        row = self.list_widget.row(item)
        if 0 <= row < len(self.items_data):
            self.sig_edit_page_style.emit(self.items_data[row])

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
        act_style = menu.addAction(get_icon("sparkles", color="#3B82F6", size=14), "🎨 修改此页文字设置...")
        act_translate = menu.addAction(get_icon("play", color="#3B82F6", size=14), "翻译此页面")
        act_export = menu.addAction(get_icon("download", color="#10B981", size=14), "导出此页面...")
        menu.addSeparator()
        act_locate = menu.addAction(get_icon("folder_open", color="#A1A1AA", size=14), "在文件资源管理器中定位")
        menu.addSeparator()
        act_clear_cache = menu.addAction(get_icon("trash", color="#F59E0B", size=14), "清除本页缓存 (.amt_cache)")
        act_remove = menu.addAction(get_icon("trash", color="#EF4444", size=14), "从列表中移除")

        chosen = menu.exec(self.list_widget.mapToGlobal(pos))
        if chosen == act_style:
            self.sig_edit_page_style.emit(data)
        elif chosen == act_translate:
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
