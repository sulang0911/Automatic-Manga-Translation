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
import logging
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from app.core.models import TranslationBlock

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = ".amt_cache"


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


def safe_cv2_imwrite(path: str, img: np.ndarray, ext: Optional[str] = None, quality: int = 95) -> bool:
    """
    Safely writes an image using OpenCV, supporting Windows Unicode and non-ASCII paths.
    If ext is not specified, infers extension from the target path (defaults to .webp).
    Uses WebP with quality 95 by default; falls back to PNG if WebP encoding fails.
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
        if not ok and is_webp:
            # Fallback to PNG
            ext = ".png"
            path = os.path.splitext(path)[0] + ".png"
            ok, buf = cv2.imencode(".png", img)
        if not ok or buf is None:
            return False
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
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

    def __init__(self, custom_cache_root: Optional[str] = None):
        self.custom_cache_root = custom_cache_root

    def get_cache_dir(self, image_path: str) -> str:
        """
        Returns the cache directory path for a given image.
        Default: `<image_parent_dir>/.amt_cache`
        """
        if self.custom_cache_root:
            rel_dir = os.path.basename(os.path.dirname(os.path.abspath(image_path)))
            cache_dir = os.path.join(self.custom_cache_root, rel_dir, CACHE_DIR_NAME)
        else:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), CACHE_DIR_NAME)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def get_cache_paths(self, image_path: str) -> Dict[str, str]:
        """
        Returns absolute paths for all cache files associated with this image.
        """
        cache_dir = self.get_cache_dir(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return {
            "dir": cache_dir,
            "erased_webp": os.path.join(cache_dir, f"{base_name}.erased.webp"),
            "erased_png": os.path.join(cache_dir, f"{base_name}.erased.png"),
            "blocks_json": os.path.join(cache_dir, f"{base_name}.blocks.json"),
            "rendered_webp": os.path.join(cache_dir, f"{base_name}.rendered.webp"),
            "rendered_png": os.path.join(cache_dir, f"{base_name}.rendered.png"),
        }

    def has_cache(self, image_path: str) -> Dict[str, bool]:
        """
        Checks which cache files exist on disk for the given image without reading them into RAM.
        """
        paths = self.get_cache_paths(image_path)
        has_erased = os.path.exists(paths["erased_webp"]) or os.path.exists(paths["erased_png"])
        has_blocks = os.path.exists(paths["blocks_json"])
        has_rendered = os.path.exists(paths["rendered_webp"]) or os.path.exists(paths["rendered_png"])
        return {
            "erased": has_erased,
            "blocks": has_blocks,
            "rendered": has_rendered,
        }

    def is_fully_translated(self, image_path: str) -> bool:
        """
        Returns True if this page has been fully translated and rendered.
        Validates that blocks.json exists, contains blocks, and at least one has translated_text.
        """
        status = self.has_cache(image_path)
        if not (status["blocks"] and (status["rendered"] or status["erased"])):
            return False
        try:
            paths = self.get_cache_paths(image_path)
            with open(paths["blocks_json"], "r", encoding="utf-8") as f:
                data = json.load(f)
            blocks_list = data if isinstance(data, list) else data.get("blocks", [])
            if not blocks_list:
                return False
            # Check if translations exist
            return any(bool(b.get("translated_text", "").strip()) for b in blocks_list if isinstance(b, dict))
        except Exception:
            return False

    def save_page_cache(
        self,
        image_path: str,
        erased_img: Optional[np.ndarray] = None,
        blocks: Optional[List[Any]] = None,
        rendered_img: Optional[np.ndarray] = None
    ) -> Dict[str, str]:
        """
        Persists intermediate translation results to disk.
        Returns a dictionary of saved file paths.
        """
        paths = self.get_cache_paths(image_path)
        saved = {}

        # 1. Save erased background
        if erased_img is not None and erased_img.size > 0:
            target_erased = paths["erased_webp"]
            if safe_cv2_imwrite(target_erased, erased_img, ext=".webp", quality=95):
                saved["erased"] = target_erased
            else:
                # Fallback to PNG
                target_erased = paths["erased_png"]
                if safe_cv2_imwrite(target_erased, erased_img, ext=".png"):
                    saved["erased"] = target_erased

        # 2. Save blocks metadata & translations
        if blocks is not None:
            serialized_blocks = []
            for b in blocks:
                if isinstance(b, TranslationBlock):
                    serialized_blocks.append(b.to_dict())
                elif isinstance(b, dict):
                    serialized_blocks.append(b)
                elif hasattr(b, "to_dict"):
                    serialized_blocks.append(b.to_dict())
                else:
                    serialized_blocks.append(vars(b))

            with open(paths["blocks_json"], "w", encoding="utf-8") as f:
                json.dump(serialized_blocks, f, ensure_ascii=False, indent=2)
            saved["blocks"] = paths["blocks_json"]

        # 3. Save rendered composite
        if rendered_img is not None and rendered_img.size > 0:
            target_rendered = paths["rendered_webp"]
            if safe_cv2_imwrite(target_rendered, rendered_img, ext=".webp", quality=95):
                saved["rendered"] = target_rendered
            else:
                target_rendered = paths["rendered_png"]
                if safe_cv2_imwrite(target_rendered, rendered_img, ext=".png"):
                    saved["rendered"] = target_rendered

        return saved

    def load_page_cache(
        self,
        image_path: str,
        load_images: bool = True
    ) -> Dict[str, Any]:
        """
        Loads cached data from disk.
        If `load_images=False`, only loads `blocks` and metadata, returning file paths
        for images without decoding large OpenCV arrays into RAM (ideal for list views).
        If `load_images=True`, decodes `erased_img` and `rendered_img` (for canvas view).
        """
        paths = self.get_cache_paths(image_path)
        res: Dict[str, Any] = {
            "blocks": [],
            "erased_img": None,
            "rendered_img": None,
            "has_cache": False,
            "erased_path": None,
            "rendered_path": None,
            "blocks_path": None,
        }

        # 1. Blocks JSON
        if os.path.exists(paths["blocks_json"]):
            try:
                with open(paths["blocks_json"], "r", encoding="utf-8") as f:
                    raw_blocks = json.load(f)
                if isinstance(raw_blocks, list):
                    res["blocks"] = [
                        TranslationBlock.from_dict(b) if isinstance(b, dict) else b
                        for b in raw_blocks
                    ]
                res["blocks_path"] = paths["blocks_json"]
                res["has_cache"] = True
            except Exception as e:
                logger.warning(f"Error loading blocks cache for {image_path}: {e}")

        # 2. Erased Image
        erased_p = paths["erased_webp"] if os.path.exists(paths["erased_webp"]) else paths["erased_png"]
        if os.path.exists(erased_p):
            res["erased_path"] = erased_p
            res["has_cache"] = True
            if load_images:
                res["erased_img"] = safe_cv2_imread(erased_p)

        # 3. Rendered Image
        rendered_p = paths["rendered_webp"] if os.path.exists(paths["rendered_webp"]) else paths["rendered_png"]
        if os.path.exists(rendered_p):
            res["rendered_path"] = rendered_p
            res["has_cache"] = True
            if load_images:
                res["rendered_img"] = safe_cv2_imread(rendered_p)

        return res

    def clear_cache(self, image_path: str) -> None:
        """Removes all cache files for a specific image."""
        paths = self.get_cache_paths(image_path)
        for key, p in paths.items():
            if key != "dir" and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {p}: {e}")

    def clear_all_cache(self, directory_path: str) -> None:
        """Removes the entire .amt_cache directory under the given folder."""
        cache_dir = os.path.join(directory_path, CACHE_DIR_NAME)
        if os.path.exists(cache_dir):
            import shutil
            try:
                shutil.rmtree(cache_dir)
            except Exception as e:
                logger.warning(f"Failed to delete cache dir {cache_dir}: {e}")


_GLOBAL_CACHE_MGR: Optional[MangaCacheManager] = None


def get_cache_manager() -> MangaCacheManager:
    """Returns the global singleton MangaCacheManager instance."""
    global _GLOBAL_CACHE_MGR
    if _GLOBAL_CACHE_MGR is None:
        _GLOBAL_CACHE_MGR = MangaCacheManager()
    return _GLOBAL_CACHE_MGR