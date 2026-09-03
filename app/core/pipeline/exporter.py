"""
app/core/pipeline/exporter.py
High-resolution multi-format exporter for Manga/Webtoon translations.
Supports PNG, JPEG, WebP with visually lossless compression, multi-page PDF compilation,
and chapter ZIP bundling conforming to F-EXP-02, F-EXP-03, and F-EXP-04.
"""
import os
import zipfile
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image

from app.ui.sidebar.page_list import natural_sort_key


class MangaExporter:
    """
    Export manager handling single page multi-format output and chapter compilation.
    """

    @staticmethod
    def export_single_image(
        image_bgr: np.ndarray,
        output_path: str,
        fmt: str = "PNG",
        compressed: bool = False
    ) -> bool:
        """
        Exports a single OpenCV BGR image matching the exact original canvas dimensions.
        """
        if image_bgr is None or image_bgr.size == 0:
            return False

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fmt = fmt.upper()

        if fmt in ["JPG", "JPEG"]:
            quality = 85 if compressed else 95
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            ext = ".jpg"
        elif fmt == "WEBP":
            quality = 80 if compressed else 92
            encode_params = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
            ext = ".webp"
        else:
            # PNG
            compression = 6 if compressed else 3
            encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), compression]
            ext = ".png"

        if not output_path.lower().endswith(ext):
            output_path += ext

        success, buf = cv2.imencode(ext, image_bgr, encode_params)
        if not success:
            return False

        with open(output_path, "wb") as f:
            f.write(buf.tobytes())
        return True

    @staticmethod
    def compile_chapter_pdf(image_paths: List[str], output_pdf_path: str) -> bool:
        """
        Compiles multiple manga pages into a single multi-page PDF in natural numeric order.
        """
        if not image_paths:
            return False

        os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)

        # Sort paths naturally: page1, page2, page10
        sorted_paths = sorted([p for p in image_paths if os.path.exists(p)], key=natural_sort_key)
        if not sorted_paths:
            return False

        pil_images = []
        for path in sorted_paths:
            try:
                img = Image.open(path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                pil_images.append(img)
            except Exception as e:
                print(f"[-] Warning: Failed to load {path} for PDF: {e}")

        if not pil_images:
            return False

        first_img = pil_images[0]
        remaining = pil_images[1:] if len(pil_images) > 1 else []
        first_img.save(
            output_pdf_path,
            "PDF",
            resolution=100.0,
            save_all=True,
            append_images=remaining
        )
        return True

    @staticmethod
    def package_chapter_zip(image_paths: List[str], output_zip_path: str) -> bool:
        """
        Bundles translated chapter images into a clean ZIP archive.
        """
        if not image_paths:
            return False

        os.makedirs(os.path.dirname(os.path.abspath(output_zip_path)), exist_ok=True)
        sorted_paths = sorted([p for p in image_paths if os.path.exists(p)], key=natural_sort_key)

        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted_paths:
                arcname = os.path.basename(p)
                zf.write(p, arcname=arcname)
        return True
