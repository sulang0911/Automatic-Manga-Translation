"""
app/core/inpaint/restore_helper.py
Pixel restoration utility for restoring original comic artwork and text
when translation blocks or speech bubbles are deleted by the user.
Includes active block collision protection to prevent overwriting adjacent inpainted bubbles.
"""
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import cv2
from app.core.models import TranslationBlock


def _extract_block_geometries(
    block: Union[Dict[str, Any], TranslationBlock],
    img_w: int,
    img_h: int
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Extracts all polygon point arrays and pixel bounding boxes associated with a block,
    including its current polygon, current normalized [xmin, ymin, xmax, ymax] box,
    any pre-drag original box/polygon (_orig_box, _orig_polygon), and any merged
    constituent boxes (_merged_boxes).
    """
    polys: List[np.ndarray] = []
    boxes: List[Tuple[int, int, int, int]] = []

    def _add_norm_box(xmin_n: float, ymin_n: float, xmax_n: float, ymax_n: float):
        x1 = max(0, min(img_w, int(round(float(xmin_n) * img_w / 100.0))))
        y1 = max(0, min(img_h, int(round(float(ymin_n) * img_h / 100.0))))
        x2 = max(0, min(img_w, int(round(float(xmax_n) * img_w / 100.0))))
        y2 = max(0, min(img_h, int(round(float(ymax_n) * img_h / 100.0))))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    def _add_poly(raw_p: Any):
        if raw_p and len(raw_p) >= 3:
            try:
                pts = np.array([[int(round(p[0])), int(round(p[1]))] for p in raw_p], dtype=np.int32)
                polys.append(pts)
                px1 = max(0, min(img_w, int(np.min(pts[:, 0]))))
                py1 = max(0, min(img_h, int(np.min(pts[:, 1]))))
                px2 = max(0, min(img_w, int(np.max(pts[:, 0]))))
                py2 = max(0, min(img_h, int(np.max(pts[:, 1]))))
                if px2 > px1 and py2 > py1:
                    boxes.append((px1, py1, px2, py2))
            except Exception:
                pass

    if isinstance(block, TranslationBlock):
        poly = block.to_pixel_polygon(img_w, img_h)
        _add_poly(poly)
        _add_norm_box(block.xmin, block.ymin, block.xmax, block.ymax)
        orig_box = getattr(block, "_orig_box", None)
        if orig_box and len(orig_box) == 4:
            _add_norm_box(orig_box[0], orig_box[1], orig_box[2], orig_box[3])
        orig_poly = getattr(block, "_orig_polygon", None)
        if orig_poly:
            _add_poly(orig_poly)
        merged_boxes = getattr(block, "_merged_boxes", None)
        if merged_boxes:
            for mb in merged_boxes:
                if isinstance(mb, dict):
                    _add_poly(mb.get("polygon"))
                    _add_norm_box(mb.get("xmin", 0.0), mb.get("ymin", 0.0), mb.get("xmax", 0.0), mb.get("ymax", 0.0))
                elif isinstance(mb, (list, tuple)) and len(mb) == 4:
                    _add_norm_box(mb[0], mb[1], mb[2], mb[3])
    elif isinstance(block, dict):
        raw_poly = block.get("polygon")
        if raw_poly and len(raw_poly) >= 3:
            _add_poly(raw_poly)
        else:
            try:
                tb = TranslationBlock.from_dict(block)
                _add_poly(tb.to_pixel_polygon(img_w, img_h))
            except Exception:
                pass
        _add_norm_box(
            block.get("xmin", 0.0),
            block.get("ymin", 0.0),
            block.get("xmax", 0.0),
            block.get("ymax", 0.0),
        )
        orig_box = block.get("_orig_box")
        if orig_box and len(orig_box) == 4:
            _add_norm_box(orig_box[0], orig_box[1], orig_box[2], orig_box[3])
        orig_poly = block.get("_orig_polygon")
        if orig_poly:
            _add_poly(orig_poly)
        merged_boxes = block.get("_merged_boxes")
        if merged_boxes:
            for mb in merged_boxes:
                if isinstance(mb, dict):
                    _add_poly(mb.get("polygon"))
                    _add_norm_box(mb.get("xmin", 0.0), mb.get("ymin", 0.0), mb.get("xmax", 0.0), mb.get("ymax", 0.0))
                elif isinstance(mb, (list, tuple)) and len(mb) == 4:
                    _add_norm_box(mb[0], mb[1], mb[2], mb[3])

    return polys, boxes


def get_block_pixel_mask(
    block: Union[Dict[str, Any], TranslationBlock],
    img_w: int,
    img_h: int,
    padding: int = 0
) -> np.ndarray:
    """
    Constructs a binary uint8 mask (shape: (img_h, img_w)) for a given translation block.
    Supports both polygon vertices and axis-aligned/rotated normalized bounding boxes.
    Applies elliptical morphological dilation if padding > 0.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if img_w <= 0 or img_h <= 0 or block is None:
        return mask

    polys, boxes = _extract_block_geometries(block, img_w, img_h)

    for poly_pts in polys:
        cv2.fillPoly(mask, [poly_pts], 255)

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    if padding > 0 and np.any(mask):
        ksize = 2 * padding + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.dilate(mask, kernel)

    return mask


def _get_core_and_expanded_inpaint_mask(
    block: Union[Dict[str, Any], TranslationBlock],
    img_w: int,
    img_h: int,
    padding: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes both the tight core mask of a block and its full expanded inpainting footprint
    (matching the adaptive expansion in erase_unboxed_text_block, InpaintEngine stroke padding,
    and Gaussian feather blending).
    """
    core = get_block_pixel_mask(block, img_w, img_h, padding=0)
    exp = np.zeros((img_h, img_w), dtype=np.uint8)
    if not np.any(core):
        return core, exp

    _, boxes = _extract_block_geometries(block, img_w, img_h)
    if not boxes:
        ys, xs = np.nonzero(core)
        boxes = [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))]

    feather_margin = max(18, int(round(max(img_w, img_h) * 0.003)) * 3 + 12)

    for (x1, y1, x2, y2) in boxes:
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        if bw >= bh:
            pad_x = max(padding, 10, int(round(bw * 0.05)), int(round(bh * 0.30))) + feather_margin
            pad_y = max(padding, 8, int(round(bh * 0.35))) + feather_margin
        else:
            pad_y = max(padding, 10, int(round(bh * 0.05)), int(round(bw * 0.30))) + feather_margin
            pad_x = max(padding, 8, int(round(bw * 0.35))) + feather_margin

        ex1 = max(0, x1 - pad_x)
        ey1 = max(0, y1 - pad_y)
        ex2 = min(img_w, x2 + pad_x)
        ey2 = min(img_h, y2 + pad_y)
        if ex2 > ex1 and ey2 > ey1:
            cv2.rectangle(exp, (ex1, ey1), (ex2, ey2), 255, -1)

    return core, exp


def restore_block_pixels(
    original_img: Optional[np.ndarray],
    erased_img: Optional[np.ndarray],
    deleted_block: Optional[Union[Dict[str, Any], TranslationBlock]],
    remaining_blocks: Optional[List[Union[Dict[str, Any], TranslationBlock]]] = None,
    padding: int = 4
) -> Optional[np.ndarray]:
    """
    Restores the original comic artwork and text pixels at the location of a deleted block.

    Args:
        original_img: Unmodified original manga scan (BGR or RGB, uint8).
        erased_img: Inpainted background image (BGR or RGB, uint8).
        deleted_block: The block being deleted (dict or TranslationBlock). Can be None if restoring all.
        remaining_blocks: List of active undeleted blocks that must remain inpainted.
        padding: Pixel padding / dilation to cover inpaint feathering edges (default: 4px).

    Returns:
        Updated erased_img with the deleted block area restored to original pixels.
    """
    if erased_img is None:
        return original_img.copy() if original_img is not None else None
    if original_img is None:
        return erased_img.copy()
    if original_img.shape[:2] != erased_img.shape[:2]:
        return erased_img.copy()

    h_img, w_img = original_img.shape[:2]

    # If no deleted block is provided and remaining_blocks is empty, restore full page to original
    if deleted_block is None and (remaining_blocks is not None and len(remaining_blocks) == 0):
        return original_img.copy()

    if deleted_block is None:
        return erased_img.copy()

    # Exact synthetic / zero-padding mode (used by unit tests with padding=0)
    if padding <= 0:
        deleted_mask = get_block_pixel_mask(deleted_block, w_img, h_img, padding=0)
        if not np.any(deleted_mask):
            return erased_img.copy()

        if remaining_blocks:
            active_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            for b in remaining_blocks:
                b_mask = get_block_pixel_mask(b, w_img, h_img, padding=1)
                if np.any(b_mask):
                    cv2.bitwise_or(active_mask, b_mask, dst=active_mask)
            restore_mask = cv2.bitwise_and(deleted_mask, cv2.bitwise_not(active_mask))
        else:
            restore_mask = deleted_mask

        restored_erased = erased_img.copy()
        restored_erased[restore_mask > 0] = original_img[restore_mask > 0]
        return restored_erased

    # Full adaptive restoration mode (padding > 0):
    # Restores the entire unboxed/stroked/feathered inpainting footprint of deleted_block
    # while protecting adjacent active bubbles via core masks and Euclidean Voronoi partitioning.
    del_core, del_exp = _get_core_and_expanded_inpaint_mask(deleted_block, w_img, h_img, padding=padding)
    if not np.any(del_exp):
        return erased_img.copy()

    if remaining_blocks:
        rem_core = np.zeros((h_img, w_img), dtype=np.uint8)
        rem_exp = np.zeros((h_img, w_img), dtype=np.uint8)
        for b in remaining_blocks:
            rc, re = _get_core_and_expanded_inpaint_mask(b, w_img, h_img, padding=padding)
            if np.any(rc):
                cv2.bitwise_or(rem_core, rc, dst=rem_core)
            if np.any(re):
                cv2.bitwise_or(rem_exp, re, dst=rem_exp)

        if np.any(rem_core):
            kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            rem_core_dilated = cv2.dilate(rem_core, kernel_3)
            # Only compute distanceTransform if del_exp actually overlaps rem_exp
            overlap = cv2.bitwise_and(del_exp, rem_exp)
            if np.any(overlap):
                dist_del = cv2.distanceTransform(cv2.bitwise_not(del_core), cv2.DIST_L2, 3)
                dist_rem = cv2.distanceTransform(cv2.bitwise_not(rem_core), cv2.DIST_L2, 3)
                restore_bool = (del_exp > 0) & (rem_core_dilated == 0) & ((rem_exp == 0) | (dist_del <= dist_rem))
            else:
                restore_bool = (del_exp > 0) & (rem_core_dilated == 0)
        else:
            restore_bool = (del_exp > 0)
    else:
        restore_bool = (del_exp > 0)

    restored_erased = erased_img.copy()
    restored_erased[restore_bool] = original_img[restore_bool]

    return restored_erased
