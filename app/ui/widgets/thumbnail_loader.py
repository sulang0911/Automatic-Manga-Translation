"""
app/ui/widgets/thumbnail_loader.py
High-performance asynchronous thumbnail manager with thread-pool offloading,
codec-level downsampled decoding (QImageReader), LIFO priority scheduling,
and true OrderedDict LRU caching to prevent GUI freezes and RAM spikes.
"""
from collections import OrderedDict
from typing import Dict, List, Tuple, Callable, Optional
import os
from PyQt6.QtCore import QObject, QThreadPool, QRunnable, pyqtSignal, Qt, QSize
from PyQt6.QtGui import QImage, QPixmap, QImageReader


def _decode_scaled_qimage(path: str, target_width: int, target_height: int) -> QImage:
    """
    Decodes an image directly into a downscaled QImage using QImageReader so that
    multi-megapixel manga pages do not allocate full-resolution buffers in RAM.
    """
    try:
        if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
            return QImage()
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        orig_size = reader.size()
        if orig_size.isValid() and orig_size.width() > 0 and orig_size.height() > 0:
            scaled_size = orig_size.scaled(
                QSize(target_width, target_height),
                Qt.AspectRatioMode.KeepAspectRatio
            )
            reader.setScaledSize(scaled_size)
            qimg = reader.read()
            if not qimg.isNull():
                return qimg
    except Exception:
        pass

    try:
        img = QImage(path)
        if not img.isNull():
            return img.scaled(
                target_width,
                target_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
    except Exception:
        pass
    return QImage()


class ThumbnailSignals(QObject):
    """Signals for background thumbnail workers."""
    loaded = pyqtSignal(str, QImage)


class ThumbnailWorker(QRunnable):
    """Background worker for reading and downscaling images without blocking the main GUI thread."""

    def __init__(
        self,
        path: str,
        target_width: int,
        target_height: int,
        signals: ThumbnailSignals,
        is_stale_fn: Optional[Callable[[], bool]] = None,
    ):
        super().__init__()
        self.path = path
        self.target_width = target_width
        self.target_height = target_height
        self.signals = signals
        self.is_stale_fn = is_stale_fn

    def run(self):
        try:
            if self.is_stale_fn is not None and self.is_stale_fn():
                return
            if not os.path.exists(self.path):
                return
            scaled = _decode_scaled_qimage(self.path, self.target_width, self.target_height)
            if self.is_stale_fn is not None and self.is_stale_fn():
                return
            if not scaled.isNull():
                try:
                    self.signals.loaded.emit(self.path, scaled)
                except RuntimeError:
                    pass
        except Exception:
            pass


class AsyncThumbnailManager:
    """
    Thread-safe asynchronous thumbnail loading manager.
    Decodes and downsamples full-resolution manga images in a dedicated lightweight
    thread pool, supporting explicit priority ordering and true OrderedDict LRU eviction.
    """
    _instance: Optional['AsyncThumbnailManager'] = None

    def __init__(self):
        self._pool = QThreadPool()
        # Thumbnail decoding uses QImageReader.setScaledSize (~7KB/thumb), so 2-4 workers stay RAM-safe
        self._pool.setMaxThreadCount(min(4, max(2, os.cpu_count() or 2)))
        self.signals = ThumbnailSignals()
        self.signals.loaded.connect(self._on_thumbnail_loaded)
        self._cache: "OrderedDict[str, QPixmap]" = OrderedDict()
        self._max_cache = 400
        self._pending_callbacks: Dict[str, List[Callable[[QPixmap], None]]] = {}
        self._priority_seq: int = 0
        self._generation: int = 0

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
        """Returns cached QPixmap if already decoded (updating LRU order), else None."""
        pix = self._cache.get(path)
        if pix is not None:
            self._cache.move_to_end(path)
        return pix

    def invalidate(self, path: str):
        """Removes a specific path from thumbnail cache to force reload."""
        self._cache.pop(path, None)
        self._pending_callbacks.pop(path, None)

    def _put_cache(self, path: str, pix: QPixmap):
        if path in self._cache:
            self._cache.move_to_end(path)
        self._cache[path] = pix
        while len(self._cache) > self._max_cache:
            self._cache.popitem(last=False)

    def _on_thumbnail_loaded(self, path: str, qimg: QImage):
        if qimg.isNull():
            self._pending_callbacks.pop(path, None)
            return
        pix = QPixmap.fromImage(qimg)
        self._put_cache(path, pix)

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
        callback: Callable[[QPixmap], None],
        sync: bool = False,
        priority: Optional[int] = None,
    ):
        """
        Fetches a downscaled thumbnail.
        If sync is True (e.g. for single-image drops), decodes immediately on caller thread.
        Otherwise offloads image decoding to the dedicated thumbnail thread pool.
        """
        if not path:
            return

        cached = self.get_cached(path)
        if cached is not None:
            callback(cached)
            return

        if sync:
            if not os.path.exists(path):
                return
            scaled = _decode_scaled_qimage(path, target_size[0], target_size[1])
            if not scaled.isNull():
                pix = QPixmap.fromImage(scaled)
                self._put_cache(path, pix)
                callback(pix)
            return

        if path in self._pending_callbacks:
            self._pending_callbacks[path].append(callback)
        else:
            self._pending_callbacks[path] = [callback]
            expected_gen = self._generation
            is_stale = lambda: (expected_gen != self._generation) or (path not in self._pending_callbacks)
            worker = ThumbnailWorker(path, target_size[0], target_size[1], self.signals, is_stale_fn=is_stale)
            if priority is None:
                self._priority_seq = (self._priority_seq + 1) % 1000000
                eff_priority = self._priority_seq
            else:
                eff_priority = int(priority)
            try:
                self._pool.start(worker, eff_priority)
            except RuntimeError:
                pass

    def clear_cache(self):
        """Clears in-memory thumbnail cache and cancels pending queued decodes."""
        self._generation += 1
        self._cache.clear()
        self._pending_callbacks.clear()
        self._priority_seq = 0

    def shutdown(self, wait_ms: int = 1000) -> None:
        """Cancels pending thumbnail decodes and waits briefly for active pool tasks."""
        self.clear_cache()
        try:
            self._pool.clear()
            self._pool.waitForDone(max(0, int(wait_ms)))
        except RuntimeError:
            pass
