"""
app/ui/widgets/thumbnail_loader.py
High-performance asynchronous thumbnail manager with thread-pool offloading
and in-memory LRU caching to prevent GUI freezes during large folder ingestion.
"""
from typing import Dict, List, Tuple, Callable, Optional
import os
from PyQt6.QtCore import QObject, QThreadPool, QRunnable, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap


class ThumbnailSignals(QObject):
    """Signals for background thumbnail workers."""
    loaded = pyqtSignal(str, QImage)


class ThumbnailWorker(QRunnable):
    """Background worker for reading and downscaling images without blocking the main GUI thread."""

    def __init__(self, path: str, target_width: int, target_height: int, signals: ThumbnailSignals):
        super().__init__()
        self.path = path
        self.target_width = target_width
        self.target_height = target_height
        self.signals = signals

    def run(self):
        try:
            if not os.path.exists(self.path):
                return
            img = QImage(self.path)
            if not img.isNull():
                scaled = img.scaled(
                    self.target_width,
                    self.target_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                try:
                    self.signals.loaded.emit(self.path, scaled)
                except RuntimeError:
                    pass
        except Exception:
            pass


class AsyncThumbnailManager:
    """
    Thread-safe asynchronous thumbnail loading manager.
    Decodes and downsamples full-resolution manga images in background threads,
    delivering lightweight QPixmap instances back to the GUI thread.
    """
    _instance: Optional['AsyncThumbnailManager'] = None

    def __init__(self):
        self.signals = ThumbnailSignals()
        self.signals.loaded.connect(self._on_thumbnail_loaded)
        self._cache: Dict[str, QPixmap] = {}
        self._max_cache = 1500
        self._pending_callbacks: Dict[str, List[Callable[[QPixmap], None]]] = {}

    @classmethod
    def instance(cls) -> 'AsyncThumbnailManager':
        if cls._instance is not None:
            try:
                # Test whether underlying C++ QObject is still alive
                cls._instance.signals.receivers(cls._instance.signals.loaded)
            except RuntimeError:
                cls._instance = None

        if cls._instance is None:
            cls._instance = AsyncThumbnailManager()
        return cls._instance

    def get_cached(self, path: str) -> Optional[QPixmap]:
        """Returns cached QPixmap if already decoded, else None."""
        return self._cache.get(path)

    def _on_thumbnail_loaded(self, path: str, qimg: QImage):
        if qimg.isNull():
            return
        pix = QPixmap.fromImage(qimg)

        # LRU eviction if cache exceeds capacity
        if len(self._cache) >= self._max_cache:
            keys_to_remove = list(self._cache.keys())[:300]
            for k in keys_to_remove:
                self._cache.pop(k, None)

        self._cache[path] = pix

        callbacks = self._pending_callbacks.pop(path, [])
        for cb in callbacks:
            try:
                cb(pix)
            except Exception:
                pass

    def request_thumbnail(
        self,
        path: str,
        target_size: Tuple[int, int],
        callback: Callable[[QPixmap], None]
    ):
        """
        Asynchronously fetches a downscaled thumbnail.
        If cached in memory, invokes callback immediately.
        """
        if not path:
            return

        if path in self._cache:
            callback(self._cache[path])
            return

        if path in self._pending_callbacks:
            self._pending_callbacks[path].append(callback)
        else:
            self._pending_callbacks[path] = [callback]
            worker = ThumbnailWorker(path, target_size[0], target_size[1], self.signals)
            try:
                QThreadPool.globalInstance().start(worker)
            except RuntimeError:
                pass

    def clear_cache(self):
        """Clears in-memory thumbnail cache."""
        self._cache.clear()
        self._pending_callbacks.clear()
