import os
import uuid
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFileDialog, QFrame, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QPixmap, QDragEnterEvent, QDropEvent
from app.ui.sidebar.page_list import is_ignored_cache_or_export, natural_sort_path_key

class QueueItemWidget(QWidget):
    def __init__(self, item_data: dict, parent=None):
        super().__init__(parent)
        self.item_data = item_data
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        # Thumbnail
        self.thumb_label = QLabel(self)
        self.thumb_label.setFixedSize(48, 48)
        self.thumb_label.setStyleSheet("background-color: #121214; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08);")
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.thumb_label)

        # Info
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(3)

        self.name_label = QLabel(os.path.basename(item_data["path"]), self)
        self.name_label.setStyleSheet("font-weight: 600; font-size: 12px; color: #F5F5F7;")
        info_layout.addWidget(self.name_label)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(6)

        self.status_badge = QLabel("待处理", self)
        self.status_badge.setProperty("class", "statusBadge statusPending")
        self.status_badge.setStyleSheet("background-color: rgba(255, 159, 10, 0.18); color: #FF9F0A; border-radius: 4px; padding: 1px 6px; font-size: 10px; font-weight: 600;")
        status_row.addWidget(self.status_badge)

        self.info_sub = QLabel(f"#{item_data.get('index', 1)}", self)
        self.info_sub.setStyleSheet("font-size: 10px; color: #8E8E93;")
        status_row.addWidget(self.info_sub)
        status_row.addStretch()

        info_layout.addLayout(status_row)
        layout.addLayout(info_layout)

        self._load_thumbnail(item_data["path"])

    def _load_thumbnail(self, path: str):
        if os.path.exists(path):
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(48, 48, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                self.thumb_label.setPixmap(scaled)

    def set_status(self, status: str, text: str = ""):
        self.item_data["status"] = status
        if status == "pending":
            self.status_badge.setText(text or "待处理")
            self.status_badge.setStyleSheet("background-color: rgba(255, 159, 10, 0.18); color: #FF9F0A; border-radius: 4px; padding: 1px 6px; font-size: 10px; font-weight: 600;")
        elif status == "processing":
            self.status_badge.setText(text or "处理中")
            self.status_badge.setStyleSheet("background-color: rgba(10, 132, 255, 0.18); color: #0A84FF; border-radius: 4px; padding: 1px 6px; font-size: 10px; font-weight: 600;")
        elif status == "completed":
            self.status_badge.setText(text or "已完成")
            self.status_badge.setStyleSheet("background-color: rgba(48, 209, 88, 0.18); color: #30D158; border-radius: 4px; padding: 1px 6px; font-size: 10px; font-weight: 600;")
        elif status == "failed":
            self.status_badge.setText(text or "失败")
            self.status_badge.setStyleSheet("background-color: rgba(255, 69, 58, 0.18); color: #FF453A; border-radius: 4px; padding: 1px 6px; font-size: 10px; font-weight: 600;")

class QueuePanel(QFrame):
    sig_image_selected = pyqtSignal(dict)
    sig_start_batch = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebarCard")
        self.setAcceptDrops(True)
        self.items_data = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 14, 12, 14)
        layout.setSpacing(10)

        # Header
        header_row = QHBoxLayout()
        title_label = QLabel("漫画页面列表", self)
        title_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #FFFFFF;")
        header_row.addWidget(title_label)

        self.count_badge = QLabel("0 页", self)
        self.count_badge.setStyleSheet("color: #8E8E93; font-size: 12px;")
        header_row.addWidget(self.count_badge)
        header_row.addStretch()

        clear_btn = QPushButton("清空", self)
        clear_btn.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        clear_btn.clicked.connect(self.clear_all)
        header_row.addWidget(clear_btn)
        layout.addLayout(header_row)

        # Action Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        add_files_btn = QPushButton("➕ 添加图片", self)
        add_files_btn.clicked.connect(self._on_add_files)
        btn_row.addWidget(add_files_btn)

        add_folder_btn = QPushButton("📁 导入文件夹", self)
        add_folder_btn.clicked.connect(self._on_add_folder)
        btn_row.addWidget(add_folder_btn)
        layout.addLayout(btn_row)

        # List Widget
        self.list_widget = QListWidget(self)
        self.list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)

        # Batch Run Button
        self.batch_btn = QPushButton("▶ 批量翻译全部页面", self)
        self.batch_btn.setObjectName("primaryBtn")
        self.batch_btn.clicked.connect(self._on_start_batch)
        layout.addWidget(self.batch_btn)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        paths = [u.toLocalFile() for u in urls if u.isLocalFile()]
        self.add_paths(paths)

    def _on_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择漫画图片", "", "图片文件 (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if files:
            self.add_paths(files)

    def _on_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择漫画章节文件夹")
        if folder:
            self.add_paths([folder])

    def add_paths(self, paths: list):
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        collected = []
        valid_input_paths = [p for p in paths if p and os.path.exists(p)]
        dir_paths = [os.path.normpath(os.path.abspath(p)) for p in valid_input_paths if os.path.isdir(p)]

        batch_root = None
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
                if ext in valid_exts:
                    collected.append({
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
                    dirs[:] = [
                        d for d in dirs
                        if not d.startswith(".") and d.lower() not in ("translation_cache", "__pycache__", ".amt_cache")
                    ]
                    for f in files:
                        full_f = os.path.normpath(os.path.join(root, f))
                        if is_ignored_cache_or_export(full_f, base_dir=effective_root):
                            continue
                        ext = os.path.splitext(f)[1].lower()
                        if ext in valid_exts:
                            collected.append({
                                "path": full_f,
                                "rel_path": os.path.relpath(full_f, effective_root),
                                "root_dir": effective_root,
                            })

        collected.sort(key=lambda it: natural_sort_path_key(it["path"]))

        for item_info in collected:
            p = item_info["path"]
            if any(it["path"] == p for it in self.items_data):
                continue
            item_data = {
                "id": str(uuid.uuid4())[:8],
                "path": p,
                "rel_path": item_info["rel_path"],
                "root_dir": item_info["root_dir"],
                "index": len(self.items_data) + 1,
                "status": "pending",
                "blocks": None,
                "erased_img": None,
                "translated_img": None
            }
            self.items_data.append(item_data)

            # Create UI Item
            list_item = QListWidgetItem(self.list_widget)
            list_item.setSizeHint(QSize(200, 60))
            widget = QueueItemWidget(item_data, self)
            list_item.setData(Qt.ItemDataRole.UserRole, item_data["id"])
            self.list_widget.addItem(list_item)
            self.list_widget.setItemWidget(list_item, widget)

        self.count_badge.setText(f"{len(self.items_data)} 页")
        if self.items_data and self.list_widget.currentRow() == -1:
            self.list_widget.setCurrentRow(0)
            self._on_item_clicked(self.list_widget.item(0))

    def _on_item_clicked(self, item: QListWidgetItem):
        img_id = item.data(Qt.ItemDataRole.UserRole)
        for it in self.items_data:
            if it["id"] == img_id:
                self.sig_image_selected.emit(it)
                break

    def update_item_status(self, img_id: str, status: str, text: str = ""):
        for idx in range(self.list_widget.count()):
            list_item = self.list_widget.item(idx)
            if list_item.data(Qt.ItemDataRole.UserRole) == img_id:
                widget = self.list_widget.itemWidget(list_item)
                if isinstance(widget, QueueItemWidget):
                    widget.set_status(status, text)
                break

    def update_item_results(self, img_id: str, results: dict):
        for it in self.items_data:
            if it["id"] == img_id:
                it["blocks"] = results.get("blocks")
                it["erased_img"] = results.get("erased_img")
                it["translated_img"] = results.get("translated_img")
                break

    def _on_start_batch(self):
        if self.items_data:
            self.sig_start_batch.emit(self.items_data)

    def clear_all(self):
        self.items_data.clear()
        self.list_widget.clear()
        self.count_badge.setText("0 页")
