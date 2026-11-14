"""
app/core/cache/cache_manager.py
Robust, low-memory file-based disk caching for manga translation workflows.
Stores intermediate results (.erased.webp, .blocks.json, .rendered.webp) in a local
.amt_cache directory to eliminate RAM bloat, enable breakpoint resumption, and provide
instant re-rendering without re-running OCR or Inpainting.
"""
from __future__ import annotations
import os
import json
import copy
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from app.core.models import TranslationBlock, StyleConfig
from app.core.config import normalize_cache_format

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = ".amt_cache"
_CACHE_IMG_EXTS = ("bmp", "webp", "png")


def safe_cv2_imread(path: str) -> Optional[np.ndarray]:
    """
    Safely reads an image using OpenCV, supporting Windows Unicode and non-ASCII paths.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data is None or len(data) == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning(f"Failed to read image from {path}: {e}")
        return None


def safe_cv2_imwrite(
    path: str,
    img: np.ndarray,
    ext: Optional[str] = None,
    quality: int = 95,
    fallback_png: bool = True,
) -> bool:
    """
    Safely writes an image using OpenCV, supporting Windows Unicode and non-ASCII paths.
    If ext is not specified, infers extension from the target path (defaults to .webp).
    Uses WebP with quality 95 by default; optionally falls back to PNG if encoding fails.
    """
    if img is None or img.size == 0:
        return False
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if ext is None:
            file_ext = os.path.splitext(path)[1].lower()
            ext = file_ext if file_ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} else ".webp"
        is_webp = ext.lower().endswith(".webp")
        is_jpg = ext.lower() in {".jpg", ".jpeg"}
        if is_webp:
            params = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        elif is_jpg:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        else:
            params = []
        ok, buf = cv2.imencode(ext, img, params)
        if not ok and fallback_png and ext.lower() != ".png":
            # Fallback to PNG
            ext = ".png"
            path = os.path.splitext(path)[0] + ".png"
            ok, buf = cv2.imencode(".png", img)
        if not ok or buf is None:
            return False
        tmp_path = f"{path}.tmp_{threading.get_ident()}"
        try:
            with open(tmp_path, "wb") as f:
                f.write(buf.tobytes())
            os.replace(tmp_path, path)
            return True
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise e
    except Exception as e:
        logger.warning(f"Failed to write image to {path}: {e}")
        return False


class MangaCacheManager:
    """
    Manages intermediate disk caching for manga translation.
    For each image `page_001.png`, creates:
      - .amt_cache/page_001.erased.webp (Inpainted background)
      - .amt_cache/page_001.blocks.json (OCR bounding boxes, texts, and translations)
      - .amt_cache/page_001.rendered.webp (Final typography composite)
    """

    def __init__(self, custom_cache_root: Optional[str] = None, cache_format: str = "bmp"):
        self.custom_cache_root = custom_cache_root
        self._cache_format = normalize_cache_format(cache_format)
        self._translated_memo: Dict[str, Tuple[float, int, bool]] = {}
        self._blocks_memo: Dict[str, Tuple[float, int, List[Dict[str, Any]]]] = {}
        self._memo_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._write_cond = threading.Condition(self._write_lock)
        self._pending_writes: Dict[str, Dict[str, Any]] = {}
        self._in_flight_key: Optional[str] = None
        self._writer_thread: Optional[threading.Thread] = None

    def set_cache_format(self, fmt: Optional[str]) -> str:
        """Sets active intermediate image format ('bmp' for fast lossless I/O, 'webp' for compact storage)."""
        self._cache_format = normalize_cache_format(fmt)
        return self._cache_format

    def get_cache_format(self) -> str:
        """Returns active intermediate image cache format ('bmp' or 'webp')."""
        return self._cache_format

    @staticmethod
    def _norm_key(image_path: str) -> str:
        return os.path.normcase(os.path.normpath(os.path.abspath(image_path)))

    @staticmethod
    def _serialize_blocks(blocks: List[Any]) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []
        for b in blocks:
            if isinstance(b, TranslationBlock):
                serialized.append(b.to_dict())
            elif isinstance(b, dict):
                serialized.append(copy.deepcopy(b))
            elif hasattr(b, "to_dict"):
                serialized.append(b.to_dict())
            else:
                serialized.append(copy.deepcopy(vars(b)))
        return serialized

    @staticmethod
    def _serialize_style(page_style: Any) -> Optional[Dict[str, Any]]:
        if page_style is None:
            return None
        if hasattr(page_style, "to_dict"):
            return page_style.to_dict()
        if isinstance(page_style, dict):
            return copy.deepcopy(page_style)
        return copy.deepcopy(vars(page_style))

    @staticmethod
    def _write_json_atomic(path: str, payload: Any) -> None:
        tmp_path = f"{path}.tmp_{threading.get_ident()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise

    def _ensure_writer_started(self) -> None:
        if self._writer_thread is None or not self._writer_thread.is_alive():
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="AMT-CacheWriter",
                daemon=True,
            )
            self._writer_thread.start()

    def _writer_loop(self) -> None:
        while True:
            with self._write_cond:
                while not self._pending_writes:
                    self._in_flight_key = None
                    self._write_cond.notify_all()
                    self._write_cond.wait()
                next_key = next(iter(self._pending_writes))
                job = self._pending_writes.pop(next_key)
                self._in_flight_key = next_key

            try:
                self._persist_payload(
                    job["image_path"],
                    erased_img=job.get("erased_img"),
                    serialized_blocks=job.get("blocks"),
                    rendered_img=job.get("rendered_img"),
                    style_dict=job.get("page_style"),
                    clear_page_style=bool(job.get("clear_page_style", False)),
                )
            except Exception as e:
                logger.warning(f"Async cache write failed for {job.get('image_path')}: {e}")
            finally:
                with self._write_cond:
                    self._in_flight_key = None
                    self._write_cond.notify_all()

    def flush_path(self, image_path: str, timeout: float = 10.0) -> bool:
        """Waits for any queued or active async cache write targeting `image_path`."""
        if not image_path:
            return True
        key = self._norm_key(image_path)
        import time
        deadline = None if timeout is None else (time.monotonic() + timeout)
        with self._write_cond:
            while key in self._pending_writes or self._in_flight_key == key:
                rem = None if deadline is None else (deadline - time.monotonic())
                if rem is not None and rem <= 0:
                    return False
                self._write_cond.wait(timeout=rem)
            return True

    def flush_pending_writes(self, timeout: float = 10.0) -> bool:
        """Blocks until all queued background cache writes finish."""
        import time
        deadline = None if timeout is None else (time.monotonic() + timeout)
        with self._write_cond:
            while self._pending_writes or self._in_flight_key is not None:
                rem = None if deadline is None else (deadline - time.monotonic())
                if rem is not None and rem <= 0:
                    return False
                self._write_cond.wait(timeout=rem)
            return True

    def get_cache_dir(self, image_path: str, create: bool = False) -> str:
        """
        Returns the cache directory path for a given image.
        Default: `<image_parent_dir>/.amt_cache`
        If create is True, ensures the directory exists on disk.
        """
        if self.custom_cache_root:
            rel_dir = os.path.basename(os.path.dirname(os.path.abspath(image_path)))
            cache_dir = os.path.join(self.custom_cache_root, rel_dir, CACHE_DIR_NAME)
        else:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), CACHE_DIR_NAME)
        if create:
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def get_cache_paths(self, image_path: str, create_dir: bool = False) -> Dict[str, str]:
        """
        Returns absolute paths for all cache files associated with this image.
        """
        cache_dir = self.get_cache_dir(image_path, create=create_dir)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return {
            "dir": cache_dir,
            "erased_bmp": os.path.join(cache_dir, f"{base_name}.erased.bmp"),
            "erased_webp": os.path.join(cache_dir, f"{base_name}.erased.webp"),
            "erased_png": os.path.join(cache_dir, f"{base_name}.erased.png"),
            "blocks_json": os.path.join(cache_dir, f"{base_name}.blocks.json"),
            "style_json": os.path.join(cache_dir, f"{base_name}.style.json"),
            "rendered_bmp": os.path.join(cache_dir, f"{base_name}.rendered.bmp"),
            "rendered_webp": os.path.join(cache_dir, f"{base_name}.rendered.webp"),
            "rendered_png": os.path.join(cache_dir, f"{base_name}.rendered.png"),
        }

    def _ordered_variant_keys(self, kind: str) -> Tuple[str, ...]:
        preferred = self._cache_format
        ordered = [preferred] + [ext for ext in _CACHE_IMG_EXTS if ext != preferred]
        return tuple(f"{kind}_{ext}" for ext in ordered)

    def resolve_cache_image_path(
        self,
        image_path: str,
        kind: str,
        paths: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Returns the existing cache image file for `kind` ('erased' or 'rendered'), prioritizing active format."""
        c_paths = paths if paths is not None else self.get_cache_paths(image_path, create_dir=False)
        for key in self._ordered_variant_keys(kind):
            cand = c_paths.get(key, "")
            if cand and os.path.exists(cand):
                return cand
        return None

    def invalidate_rendered_cache(self, image_path: str) -> None:
        """Removes all rendered bitmap variants for `image_path` while preserving erased background and blocks."""
        c_paths = self.get_cache_paths(image_path, create_dir=False)
        for ext in _CACHE_IMG_EXTS:
            cand = c_paths.get(f"rendered_{ext}", "")
            if cand and os.path.exists(cand):
                try:
                    os.remove(cand)
                except OSError:
                    pass

    def has_cache(self, image_path: str) -> Dict[str, bool]:
        """
        Checks which cache files exist on disk for the given image without reading them into RAM.
        Fast-prunes checks if .amt_cache directory does not exist.
        """
        self.flush_path(image_path)
        cache_dir = self.get_cache_dir(image_path, create=False)
        if not os.path.isdir(cache_dir):
            return {
                "erased": False,
                "blocks": False,
                "rendered": False,
            }
        paths = self.get_cache_paths(image_path, create_dir=False)
        has_erased = any(os.path.exists(paths[f"erased_{ext}"]) for ext in _CACHE_IMG_EXTS)
        has_blocks = os.path.exists(paths["blocks_json"])
        has_rendered = any(os.path.exists(paths[f"rendered_{ext}"]) for ext in _CACHE_IMG_EXTS)
        return {
            "erased": has_erased,
            "blocks": has_blocks,
            "rendered": has_rendered,
        }

    def _check_translated_blocks_file(self, b_path: str) -> bool:
        """Checks whether a blocks.json file contains non-empty translated text, memoized by (mtime, size)."""
        try:
            st = os.stat(b_path)
            with self._memo_lock:
                memo = self._translated_memo.get(b_path)
                if memo is not None and memo[0] == st.st_mtime and memo[1] == st.st_size:
                    return memo[2]
            with open(b_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            blocks_list = data if isinstance(data, list) else data.get("blocks", [])
            if not blocks_list:
                res = False
            else:
                res = any(bool(b.get("translated_text", "").strip()) for b in blocks_list if isinstance(b, dict))
            with self._memo_lock:
                if len(self._translated_memo) >= 1000:
                    self._translated_memo.clear()
                self._translated_memo[b_path] = (st.st_mtime, st.st_size, res)
                if isinstance(blocks_list, list):
                    if len(self._blocks_memo) >= 500:
                        self._blocks_memo.clear()
                    self._blocks_memo[b_path] = (st.st_mtime, st.st_size, blocks_list)
            return res
        except Exception:
            return False

    def is_fully_translated(self, image_path: str) -> bool:
        """
        Returns True if this page has been fully translated and rendered.
        Validates that blocks.json exists, contains blocks, and at least one has translated_text.
        Uses (mtime, size) memoization to avoid repeated disk JSON parsing during folder ingestion.
        """
        status = self.has_cache(image_path)
        if not (status["blocks"] and (status["rendered"] or status["erased"])):
            return False
        paths = self.get_cache_paths(image_path)
        return self._check_translated_blocks_file(paths["blocks_json"])

    def batch_cache_status(self, image_paths: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Batch-queries cache presence, translation completion, and saved page style for many pages
        using one os.scandir() per unique .amt_cache directory instead of per-file syscalls.
        """
        results: Dict[str, Dict[str, Any]] = {}
        if not image_paths:
            return results

        # Flush only paths that currently have pending async writes
        with self._write_lock:
            active_keys = set(self._pending_writes.keys())
            if self._in_flight_key is not None:
                active_keys.add(self._in_flight_key)
        if active_keys:
            for p in image_paths:
                if self._norm_key(p) in active_keys:
                    self.flush_path(p)

        dir_entries_cache: Dict[str, set] = {}
        for image_path in image_paths:
            cache_dir = self.get_cache_dir(image_path, create=False)
            dir_norm = os.path.normcase(os.path.normpath(cache_dir))
            files_in_dir = dir_entries_cache.get(dir_norm)
            if files_in_dir is None:
                try:
                    with os.scandir(cache_dir) as it:
                        files_in_dir = {entry.name for entry in it if entry.is_file()}
                except OSError:
                    files_in_dir = set()
                dir_entries_cache[dir_norm] = files_in_dir

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            has_erased = any(f"{base_name}.erased.{ext}" in files_in_dir for ext in _CACHE_IMG_EXTS)
            has_blocks = f"{base_name}.blocks.json" in files_in_dir
            has_rendered = any(f"{base_name}.rendered.{ext}" in files_in_dir for ext in _CACHE_IMG_EXTS)
            has_style = f"{base_name}.style.json" in files_in_dir

            fully_translated = False
            if has_blocks and (has_rendered or has_erased):
                fully_translated = self._check_translated_blocks_file(
                    os.path.join(cache_dir, f"{base_name}.blocks.json")
                )

            page_style = None
            if has_style:
                try:
                    with open(os.path.join(cache_dir, f"{base_name}.style.json"), "r", encoding="utf-8") as sf:
                        page_style = json.load(sf)
                except Exception:
                    page_style = None

            results[image_path] = {
                "erased": has_erased,
                "blocks": has_blocks,
                "rendered": has_rendered,
                "style": has_style,
                "fully_translated": fully_translated,
                "page_style": page_style,
            }
        return results

    def _save_cache_image(self, kind: str, img: Optional[np.ndarray], paths: Dict[str, str]) -> Optional[str]:
        """Persists a cache image in active format (fallback PNG) and removes stale alternate formats."""
        if img is None or img.size == 0:
            return None
        fmt = self._cache_format
        target_path = paths[f"{kind}_{fmt}"]
        if not safe_cv2_imwrite(target_path, img, ext=f".{fmt}", quality=95, fallback_png=False):
            target_path = paths[f"{kind}_png"]
            if not safe_cv2_imwrite(target_path, img, ext=".png"):
                return None
        for ext in _CACHE_IMG_EXTS:
            alt_path = paths.get(f"{kind}_{ext}")
            if alt_path and alt_path != target_path and os.path.exists(alt_path):
                try:
                    os.remove(alt_path)
                except OSError:
                    pass
        return target_path

    def _persist_payload(
        self,
        image_path: str,
        erased_img: Optional[np.ndarray] = None,
        serialized_blocks: Optional[List[Dict[str, Any]]] = None,
        rendered_img: Optional[np.ndarray] = None,
        style_dict: Optional[Dict[str, Any]] = None,
        clear_page_style: bool = False,
    ) -> Dict[str, str]:
        paths = self.get_cache_paths(image_path, create_dir=True)
        saved: Dict[str, str] = {}

        # 1. Save erased background
        saved_erased = self._save_cache_image("erased", erased_img, paths)
        if saved_erased:
            saved["erased"] = saved_erased

        # 2. Save blocks metadata & translations
        if serialized_blocks is not None:
            self._write_json_atomic(paths["blocks_json"], serialized_blocks)
            saved["blocks"] = paths["blocks_json"]
            try:
                st = os.stat(paths["blocks_json"])
                has_tr = any(bool(b.get("translated_text", "").strip()) for b in serialized_blocks if isinstance(b, dict))
                with self._memo_lock:
                    self._translated_memo[paths["blocks_json"]] = (st.st_mtime, st.st_size, has_tr)
                    self._blocks_memo[paths["blocks_json"]] = (st.st_mtime, st.st_size, serialized_blocks)
            except Exception:
                with self._memo_lock:
                    self._translated_memo.pop(paths["blocks_json"], None)
                    self._blocks_memo.pop(paths["blocks_json"], None)

        # 3. Save rendered composite
        saved_rendered = self._save_cache_image("rendered", rendered_img, paths)
        if saved_rendered:
            saved["rendered"] = saved_rendered

        # 4. Save or clear per-page style configuration
        if clear_page_style:
            if os.path.exists(paths["style_json"]):
                try:
                    os.remove(paths["style_json"])
                except Exception:
                    pass
        elif style_dict is not None:
            try:
                self._write_json_atomic(paths["style_json"], style_dict)
                saved["style"] = paths["style_json"]
            except Exception as e:
                logger.warning(f"Failed to save page style for {image_path}: {e}")

        return saved

    def save_page_cache(
        self,
        image_path: str,
        erased_img: Optional[np.ndarray] = None,
        blocks: Optional[List[Any]] = None,
        rendered_img: Optional[np.ndarray] = None,
        page_style: Optional[Any] = None,
        clear_page_style: bool = False
    ) -> Dict[str, str]:
        """
        Persists intermediate translation results to disk.
        Returns a dictionary of saved file paths.
        """
        key = self._norm_key(image_path)
        with self._write_cond:
            pending = self._pending_writes.pop(key, None)
        self.flush_path(image_path)
        serialized_blocks = self._serialize_blocks(blocks) if blocks is not None else None
        style_dict = self._serialize_style(page_style) if page_style is not None else None
        if pending is not None:
            if erased_img is None:
                erased_img = pending.get("erased_img")
            if serialized_blocks is None:
                serialized_blocks = pending.get("blocks")
            if rendered_img is None:
                rendered_img = pending.get("rendered_img")
            if not clear_page_style and style_dict is None:
                if pending.get("clear_page_style"):
                    clear_page_style = True
                else:
                    style_dict = pending.get("page_style")
        return self._persist_payload(
            image_path,
            erased_img=erased_img,
            serialized_blocks=serialized_blocks,
            rendered_img=rendered_img,
            style_dict=style_dict,
            clear_page_style=clear_page_style,
        )

    def save_page_cache_async(
        self,
        image_path: str,
        erased_img: Optional[np.ndarray] = None,
        blocks: Optional[List[Any]] = None,
        rendered_img: Optional[np.ndarray] = None,
        page_style: Optional[Any] = None,
        clear_page_style: bool = False,
    ) -> None:
        """
        Queues intermediate translation results for coalesced background disk encoding.
        Repeated edits for the same page supersede queued pending WebP/JSON payloads.
        """
        if not image_path:
            return
        serialized_blocks = self._serialize_blocks(blocks) if blocks is not None else None
        style_dict = self._serialize_style(page_style) if page_style is not None else None
        key = self._norm_key(image_path)

        with self._write_cond:
            existing = self._pending_writes.get(key)
            if existing is None:
                self._pending_writes[key] = {
                    "image_path": image_path,
                    "erased_img": erased_img,
                    "blocks": serialized_blocks,
                    "rendered_img": rendered_img,
                    "page_style": style_dict,
                    "clear_page_style": clear_page_style,
                }
            else:
                if erased_img is not None:
                    existing["erased_img"] = erased_img
                if serialized_blocks is not None:
                    existing["blocks"] = serialized_blocks
                if rendered_img is not None:
                    existing["rendered_img"] = rendered_img
                if clear_page_style:
                    existing["clear_page_style"] = True
                    existing["page_style"] = None
                elif style_dict is not None:
                    existing["clear_page_style"] = False
                    existing["page_style"] = style_dict
            self._ensure_writer_started()
            self._write_cond.notify_all()

    def load_page_cache(
        self,
        image_path: str,
        load_images: bool = True,
        load_erased: bool = True,
        load_rendered: bool = True
    ) -> Dict[str, Any]:
        """
        Loads cached data from disk.
        If `load_images=False`, only loads `blocks` and metadata, returning file paths
        for images without decoding large OpenCV arrays into RAM (ideal for list views).
        If `load_images=True`, decodes `erased_img` (when load_erased=True) and
        `rendered_img` (when load_rendered=True).
        """
        self.flush_path(image_path)
        paths = self.get_cache_paths(image_path)
        res: Dict[str, Any] = {
            "blocks": [],
            "erased_img": None,
            "rendered_img": None,
            "page_style": None,
            "has_cache": False,
            "erased_path": None,
            "rendered_path": None,
            "blocks_path": None,
        }
        if os.path.exists(paths["style_json"]):
            try:
                with open(paths["style_json"], "r", encoding="utf-8") as f:
                    raw_style = json.load(f)
                if isinstance(raw_style, dict):
                    res["page_style"] = StyleConfig.from_dict(raw_style)
            except Exception:
                pass

        # 1. Blocks JSON
        b_path = paths["blocks_json"]
        if os.path.exists(b_path):
            try:
                st = os.stat(b_path)
                with self._memo_lock:
                    memo_b = self._blocks_memo.get(b_path)
                if memo_b is not None and memo_b[0] == st.st_mtime and memo_b[1] == st.st_size:
                    raw_blocks = memo_b[2]
                else:
                    with open(b_path, "r", encoding="utf-8") as f:
                        raw_blocks = json.load(f)
                    if isinstance(raw_blocks, list):
                        with self._memo_lock:
                            if len(self._blocks_memo) >= 500:
                                self._blocks_memo.clear()
                            self._blocks_memo[b_path] = (st.st_mtime, st.st_size, raw_blocks)
                if isinstance(raw_blocks, list):
                    res["blocks"] = [
                        TranslationBlock.from_dict(b) if isinstance(b, dict) else b
                        for b in raw_blocks
                    ]
                res["blocks_path"] = b_path
                res["has_cache"] = True
            except Exception as e:
                logger.warning(f"Error loading blocks cache for {image_path}: {e}")

        # 2. Erased Image
        erased_p = self.resolve_cache_image_path(image_path, "erased", paths=paths)
        if erased_p:
            res["erased_path"] = erased_p
            res["has_cache"] = True
            if load_images and load_erased:
                res["erased_img"] = safe_cv2_imread(erased_p)

        # 3. Rendered Image
        rendered_p = self.resolve_cache_image_path(image_path, "rendered", paths=paths)
        if rendered_p:
            res["rendered_path"] = rendered_p
            res["has_cache"] = True
            if load_images and load_rendered:
                res["rendered_img"] = safe_cv2_imread(rendered_p)

        return res

    def clear_cache(self, image_path: str) -> None:
        """Removes all cache files for a specific image."""
        key = self._norm_key(image_path)
        with self._write_cond:
            self._pending_writes.pop(key, None)
        self.flush_path(image_path)
        paths = self.get_cache_paths(image_path)
        with self._memo_lock:
            self._translated_memo.pop(paths["blocks_json"], None)
            self._blocks_memo.pop(paths["blocks_json"], None)
        for pkey, p in paths.items():
            if pkey != "dir" and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {p}: {e}")

    def clear_all_cache(self, directory_path: str) -> None:
        """Removes the entire .amt_cache directory under the given folder."""
        cache_dir = os.path.join(directory_path, CACHE_DIR_NAME)
        norm_cdir = self._norm_key(cache_dir) + os.sep
        with self._memo_lock:
            for k in [k for k in self._translated_memo if self._norm_key(k).startswith(norm_cdir)]:
                self._translated_memo.pop(k, None)
            for k in [k for k in self._blocks_memo if self._norm_key(k).startswith(norm_cdir)]:
                self._blocks_memo.pop(k, None)
        if os.path.exists(cache_dir):
            import shutil
            try:
                shutil.rmtree(cache_dir)
            except Exception as e:
                logger.warning(f"Failed to delete cache dir {cache_dir}: {e}")

    def clear_caches_for_items(self, items_or_paths: List[Any]) -> None:
        """
        Removes software-generated cache files for the given items or file paths,
        deleting empty .amt_cache folders afterward without wiping sibling items' caches.
        """
        cache_dirs = set()
        paths = []
        for it in items_or_paths:
            if isinstance(it, dict):
                p = it.get("path")
            else:
                p = str(it)
            if p:
                paths.append(p)
                cache_dirs.add(self.get_cache_dir(p, create=False))

        for p in paths:
            self.clear_cache(p)

        for cdir in cache_dirs:
            if os.path.isdir(cdir):
                try:
                    if not os.listdir(cdir):
                        os.rmdir(cdir)
                except OSError as e:
                    logger.warning(f"Failed to remove empty cache dir {cdir}: {e}")


_GLOBAL_CACHE_MGR: Optional[MangaCacheManager] = None


def get_cache_manager() -> MangaCacheManager:
    """Returns the global singleton MangaCacheManager instance."""
    global _GLOBAL_CACHE_MGR
    if _GLOBAL_CACHE_MGR is None:
        _GLOBAL_CACHE_MGR = MangaCacheManager()
    return _GLOBAL_CACHE_MGR