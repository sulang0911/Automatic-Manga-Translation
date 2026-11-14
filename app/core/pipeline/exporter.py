"""
app/core/pipeline/exporter.py
High-resolution multi-format exporter for Manga/Webtoon translations.
Supports PNG, JPEG, WebP with visually lossless compression, multi-page PDF compilation,
and chapter ZIP bundling conforming to F-EXP-02, F-EXP-03, and F-EXP-04.
"""
import os
import zipfile
from typing import Any, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from app.ui.sidebar.page_list import natural_sort_key


class MangaExporter:
    """
    Export manager handling single page multi-format output and chapter compilation.
    """

    _EXT_TO_FMT = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".webp": "WEBP",
        ".bmp": "BMP",
    }

    @classmethod
    def resolve_export_format(cls, path: str, compressed: bool = False) -> Tuple[str, str]:
        """Maps an image path extension to (normalized_ext, export_fmt); forces WebP when compressed=True."""
        if compressed:
            return ".webp", "WEBP"
        ext = os.path.splitext(path)[1].lower()
        fmt = cls._EXT_TO_FMT.get(ext, "PNG")
        norm_ext = ext if ext in cls._EXT_TO_FMT else ".png"
        return norm_ext, fmt

    @staticmethod
    def is_export_compressed(config: Any) -> bool:
        """Extracts style.export_compressed from an AppConfig, StyleConfig, or dict."""
        if hasattr(config, "style"):
            return bool(getattr(config.style, "export_compressed", False))
        if hasattr(config, "export_compressed"):
            return bool(getattr(config, "export_compressed", False))
        if isinstance(config, dict):
            style_cfg = config.get("style", config)
            if isinstance(style_cfg, dict):
                return bool(style_cfg.get("export_compressed", False))
            return bool(getattr(style_cfg, "export_compressed", False))
        return False

    @staticmethod
    def compute_export_path(
        image_path: str,
        export_dir: str,
        rel_path: Optional[str] = None,
        root_dir: Optional[str] = None,
        compressed: bool = False,
    ) -> str:
        """
        Computes the target export destination path, strictly preserving the relative
        subfolder directory hierarchy 1:1 and original filename (or `.webp` when compressed=True).

        Guarantees that the export destination is separate from the original source
        directory and source file to prevent overwriting or source folder contamination.
        """
        abs_img = os.path.normpath(os.path.abspath(image_path))
        abs_export_dir = os.path.normpath(os.path.abspath(export_dir))

        norm_img = os.path.normcase(os.path.realpath(abs_img))
        norm_export = os.path.normcase(os.path.realpath(abs_export_dir))

        if root_dir:
            abs_root = os.path.normpath(os.path.abspath(root_dir))
            norm_root = os.path.normcase(os.path.realpath(abs_root))
            if norm_root == norm_export:
                raise ValueError(
                    f"Export directory '{abs_export_dir}' cannot be the same as the source root directory '{abs_root}'! "
                    f"Original source files and folder structure must remain untouched."
                )

        norm_rel = None
        if root_dir:
            abs_root = os.path.normpath(os.path.abspath(root_dir))
            try:
                computed_rel = os.path.relpath(abs_img, abs_root)
                if not computed_rel.startswith("..") and not os.path.isabs(computed_rel):
                    norm_rel = computed_rel
            except ValueError:
                pass

        if not norm_rel and rel_path:
            clean_rel = os.path.normpath(rel_path).lstrip(os.sep).lstrip("/").lstrip("\\")
            if not clean_rel.startswith("..") and not os.path.isabs(clean_rel):
                norm_rel = clean_rel

        if not norm_rel:
            norm_rel = os.path.basename(abs_img)

        target_path = os.path.normpath(os.path.join(abs_export_dir, norm_rel))
        norm_target = os.path.normcase(os.path.realpath(target_path))

        # CRITICAL RESTRICTION: Translated files MUST NOT be saved into original source folder/file!
        if norm_target == norm_img:
            raise ValueError(
                f"Export path '{target_path}' matches original source file! "
                f"Export destination folder must be independent and separate from the source folder."
            )

        if not root_dir:
            img_dir = os.path.normcase(os.path.realpath(os.path.dirname(abs_img)))
            if norm_target == os.path.normcase(os.path.realpath(os.path.join(img_dir, os.path.basename(abs_img)))):
                raise ValueError(
                    f"Export path '{target_path}' matches original source file! "
                    f"Export destination folder must be independent and separate from the source folder."
                )

        if compressed:
            target_path = os.path.splitext(target_path)[0] + ".webp"
            if os.path.normcase(os.path.realpath(target_path)) == norm_img:
                raise ValueError(
                    f"Export path '{target_path}' matches original source file! "
                    f"Export destination folder must be independent and separate from the source folder."
                )

        return target_path

    @classmethod
    def export_hierarchical_image(
        cls,
        image_bgr: np.ndarray,
        export_path: str,
        source_path: Optional[str] = None,
        fmt: Optional[str] = None,
        compressed: bool = False
    ) -> bool:
        """
        Exports a single translated image to export_path preserving original dimensions and format
        (fixed to WebP when compressed=True and fmt is omitted).
        Validates that source_path is not overwritten.
        """
        if image_bgr is None or image_bgr.size == 0:
            return False

        abs_export = os.path.normpath(os.path.abspath(export_path))
        if not fmt:
            norm_ext, fmt = cls.resolve_export_format(abs_export, compressed=compressed)
            if compressed and os.path.splitext(abs_export)[1].lower() != norm_ext:
                abs_export = os.path.splitext(abs_export)[0] + norm_ext

        if source_path:
            abs_src = os.path.normpath(os.path.abspath(source_path))
            if os.path.normcase(os.path.realpath(abs_export)) == os.path.normcase(os.path.realpath(abs_src)):
                raise ValueError(f"Cannot overwrite original source image: {abs_src}")

        os.makedirs(os.path.dirname(abs_export), exist_ok=True)
        return cls.export_single_image(image_bgr, abs_export, fmt=fmt, compressed=compressed)

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
            ext = ".jpeg" if output_path.lower().endswith(".jpeg") else ".jpg"
            if not output_path.lower().endswith((".jpg", ".jpeg")):
                output_path += ext
        elif fmt == "WEBP":
            quality = 80 if compressed else 92
            encode_params = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
            ext = ".webp"
            if not output_path.lower().endswith(".webp"):
                output_path += ext
        elif fmt == "BMP":
            encode_params = []
            ext = ".bmp"
            if not output_path.lower().endswith(".bmp"):
                output_path += ext
        else:
            # PNG: fast lossless compression by default; higher compression when export_compressed=True
            compression = 6 if compressed else 1
            encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), compression]
            ext = ".png"
            if not output_path.lower().endswith(".png"):
                output_path += ext

        success, buf = cv2.imencode(ext, image_bgr, encode_params)
        if not success:
            return False

        with open(output_path, "wb") as f:
            f.write(buf.tobytes())
        return True

    @staticmethod
    def export_single_image_pdf(image_bgr: np.ndarray, output_pdf_path: str) -> bool:
        """
        Exports a single OpenCV BGR image directly to a PDF document in memory
        without writing temporary intermediate PNG files to disk.
        """
        if image_bgr is None or image_bgr.size == 0:
            return False

        os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        try:
            pil_img.save(output_pdf_path, "PDF", resolution=100.0)
            return os.path.exists(output_pdf_path)
        finally:
            try:
                pil_img.close()
            except Exception:
                pass
            del pil_img, rgb

    @staticmethod
    def compile_chapter_pdf(image_paths: List[str], output_pdf_path: str) -> bool:
        """
        Compiles multiple manga pages into a single multi-page PDF in natural numeric order.
        Uses PyMuPDF streaming insertion (O(1) RAM per page + raw PNG/JPEG passthrough)
        when available, falling back to Pillow multi-page PDF encoding otherwise.
        """
        if not image_paths:
            return False

        os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)

        # Sort paths naturally: page1, page2, page10
        sorted_paths = sorted([p for p in image_paths if os.path.exists(p)], key=natural_sort_key)
        if not sorted_paths:
            return False

        try:
            import io
            import pymupdf

            doc = pymupdf.open()
            try:
                for path in sorted_paths:
                    try:
                        img_bytes: Optional[bytes] = None
                        w_px = h_px = 0
                        ext = os.path.splitext(path)[1].lower()
                        with Image.open(path) as im:
                            w_px, h_px = im.size
                            if ext in (".png", ".jpg", ".jpeg") and im.mode in ("RGB", "L"):
                                with open(path, "rb") as f_raw:
                                    img_bytes = f_raw.read()
                            else:
                                buf = io.BytesIO()
                                rgb_im = im.convert("RGB") if im.mode != "RGB" else im
                                try:
                                    rgb_im.save(buf, format="JPEG", quality=95)
                                finally:
                                    if rgb_im is not im:
                                        rgb_im.close()
                                img_bytes = buf.getvalue()

                        if img_bytes and w_px > 0 and h_px > 0:
                            page = doc.new_page(
                                width=w_px * 72.0 / 100.0,
                                height=h_px * 72.0 / 100.0,
                            )
                            page.insert_image(page.rect, stream=img_bytes)
                    except Exception as e:
                        print(f"[-] Warning: Failed to load {path} for PDF: {e}")

                if doc.page_count == 0:
                    return False

                doc.save(output_pdf_path, garbage=1, deflate=True)
                return True
            finally:
                doc.close()
                try:
                    from app.core.hardware import cleanup_gpu_memory
                    cleanup_gpu_memory(trim_ram=True)
                except Exception:
                    pass
        except ImportError:
            pass

        pil_images = []
        raw_handles = []
        try:
            for path in sorted_paths:
                try:
                    raw_im = Image.open(path)
                    raw_handles.append(raw_im)
                    if raw_im.mode != "RGB":
                        rgb_im = raw_im.convert("RGB")
                        raw_im.close()
                        pil_images.append(rgb_im)
                    else:
                        raw_im.load()
                        pil_images.append(raw_im)
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
        finally:
            for im in pil_images:
                try:
                    im.close()
                except Exception:
                    pass
            for im in raw_handles:
                try:
                    im.close()
                except Exception:
                    pass
            pil_images.clear()
            raw_handles.clear()
            try:
                from app.core.hardware import cleanup_gpu_memory
                cleanup_gpu_memory(trim_ram=True)
            except Exception:
                pass

    @staticmethod
    def package_chapter_zip(image_paths: List[str], output_zip_path: str) -> bool:
        """
        Bundles translated chapter images into a clean ZIP archive.
        """
        if not image_paths:
            return False

        os.makedirs(os.path.dirname(os.path.abspath(output_zip_path)), exist_ok=True)
        sorted_paths = sorted([p for p in image_paths if os.path.exists(p)], key=natural_sort_key)

        compressed_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif"}
        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted_paths:
                arcname = os.path.basename(p)
                ext = os.path.splitext(p)[1].lower()
                # Skip futile DEFLATE CPU work on already-compressed image formats
                comp = zipfile.ZIP_STORED if ext in compressed_exts else zipfile.ZIP_DEFLATED
                zf.write(p, arcname=arcname, compress_type=comp)
        return True
