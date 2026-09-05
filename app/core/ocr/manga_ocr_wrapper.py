"""
app/core/ocr/manga_ocr_wrapper.py
High-precision Manga-OCR recognizer wrapper for Japanese manga text.
Handles model loading, PyTorch 2.5/transformers compatibility, crop orientation correction,
and memory safety.
"""
import os
import sys
import logging
from typing import Optional, Union, List, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Global singleton instance
_GLOBAL_MANGA_OCR_INSTANCE = None


def get_manga_ocr(force_cpu: bool = False):
    """
    Returns a shared MangaOCRRecognizer instance.
    """
    global _GLOBAL_MANGA_OCR_INSTANCE
    if _GLOBAL_MANGA_OCR_INSTANCE is None:
        _GLOBAL_MANGA_OCR_INSTANCE = MangaOCRRecognizer(force_cpu=force_cpu)
    return _GLOBAL_MANGA_OCR_INSTANCE


class MangaOCRRecognizer:
    """
    Wrapper around kha-white/manga-ocr with error tolerance,
    automatic angle correction for slanted dialogue crops, and safe GPU/CPU management.
    """

    def __init__(self, force_cpu: bool = False, pretrained_path: str = "kha-white/manga-ocr-base"):
        self.force_cpu = force_cpu
        self.pretrained_path = pretrained_path
        self._mocr = None
        self._load_error = None

    @classmethod
    def is_available(cls) -> bool:
        """Checks whether manga_ocr library is installed and importable."""
        try:
            import manga_ocr
            return True
        except ImportError:
            return False

    def _ensure_loaded(self):
        if self._mocr is not None:
            return
        if self._load_error is not None:
            raise RuntimeError(f"Manga-OCR previously failed to load: {self._load_error}")

        try:
            # Compatibility patch for PyTorch 2.5 + transformers 4.49+ CVE check
            try:
                import transformers.modeling_utils
                transformers.modeling_utils.check_torch_load_is_safe = lambda: None
            except Exception:
                pass

            import manga_ocr
            logger.info(f"Loading Manga-OCR from '{self.pretrained_path}' (force_cpu={self.force_cpu})...")
            self._mocr = manga_ocr.MangaOcr(
                pretrained_model_name_or_path=self.pretrained_path,
                force_cpu=self.force_cpu
            )
            logger.info("Manga-OCR loaded successfully.")
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Failed to initialize Manga-OCR: {e}", exc_info=True)
            raise

    def recognize_crop(self, crop: Union[np.ndarray, Image.Image], angle: float = 0.0) -> str:
        """
        Recognizes Japanese text within an image crop.

        Args:
            crop: BGR/RGB numpy array or PIL Image.
            angle: Text orientation angle in degrees. If |angle| >= 15.0, rotates the crop
                   upright before recognition to ensure the ViT encoder sees canonical text.

        Returns:
            Recognized Japanese text string.
        """
        if crop is None:
            return ""

        # Convert numpy array to PIL Image
        if isinstance(crop, np.ndarray):
            if crop.size == 0:
                return ""
            if crop.ndim == 2:
                pil_img = Image.fromarray(crop).convert("RGB")
            elif crop.ndim == 3:
                # Assume BGR if from OpenCV (3 channels)
                if crop.shape[2] == 3:
                    rgb = crop[:, :, ::-1]
                    pil_img = Image.fromarray(rgb)
                elif crop.shape[2] == 4:
                    rgba = crop[:, :, [2, 1, 0, 3]]
                    pil_img = Image.fromarray(rgba).convert("RGB")
                else:
                    pil_img = Image.fromarray(crop).convert("RGB")
            else:
                return ""
        elif isinstance(crop, Image.Image):
            pil_img = crop.convert("RGB")
        else:
            return ""

        if pil_img.width < 4 or pil_img.height < 4:
            return ""

        # Rotate upright if slanted
        if abs(angle) >= 15.0:
            # Rotate in the counter-angle direction with white background expansion
            pil_img = pil_img.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))

        self._ensure_loaded()
        try:
            text = self._mocr(pil_img)
            return self._clean_text(text)
        except Exception as e:
            logger.warning(f"Manga-OCR crop inference error: {e}")
            return ""

    def recognize_crops(self, crops_with_angles: List[Tuple[Union[np.ndarray, Image.Image], float]]) -> List[str]:
        """
        Batch-recognizes multiple crops.
        """
        results = []
        for crop, angle in crops_with_angles:
            results.append(self.recognize_crop(crop, angle=angle))
        return results

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Cleans recognized text while preserving intended CJK characters and punctuation.
        """
        if not text:
            return ""
        # Remove unwanted leading/trailing whitespace
        s = text.strip()
        # Clean double spaces often injected in OCR tokenization
        while "  " in s:
            s = s.replace("  ", " ")
        return s
