"""
app/core/inpaint/restore_helper.py
Pixel restoration utility for restoring original comic artwork and text
when translation blocks or speech bubbles are deleted by the user.
Includes active block collision protection to prevent overwriting adjacent inpainted bubbles.
"""
from typing import List, Dict, Any, Optional, Union
import numpy as np
import cv2
from app.core.models import TranslationBlock


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

    poly = None
    if isinstance(block, TranslationBlock):
        poly = block.to_pixel_polygon(img_w, img_h)
    elif isinstance(block, dict):
        raw_poly = block.get("polygon")
        if raw_poly and len(raw_poly) >= 3:
            poly = [[int(round(p[0])), int(round(p[1]))] for p in raw_poly]
        else:
            try:
                tb = TranslationBlock.from_dict(block)
                poly = tb.to_pixel_polygon(img_w, img_h)
            except Exception:
                xmin = max(0, min(img_w, int(round(block.get("xmin", 0.0) * img_w / 100.0))))
                ymin = max(0, min(img_h, int(round(block.get("ymin", 0.0) * img_h / 100.0))))
                xmax = max(0, min(img_w, int(round(block.get("xmax", 0.0) * img_w / 100.0))))
                ymax = max(0, min(img_h, int(round(block.get("ymax", 0.0) * img_h / 100.0))))
                if xmax > xmin and ymax > ymin:
                    poly = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

    if poly and len(poly) >= 3:
        poly_pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask, [poly_pts], 255)
        if padding > 0:
            ksize = 2 * padding + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            mask = cv2.dilate(mask, kernel)

    return mask


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

    # 1. Generate mask for the deleted block (with padding to cover feathering boundaries)
    deleted_mask = get_block_pixel_mask(deleted_block, w_img, h_img, padding=padding)
    if np.sum(deleted_mask) == 0:
        return erased_img.copy()

    # 2. Collision Protection: subtract remaining active blocks so adjacent active bubbles stay inpainted
    if remaining_blocks:
        active_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        for b in remaining_blocks:
            b_mask = get_block_pixel_mask(b, w_img, h_img, padding=1)
            if np.sum(b_mask) > 0:
                cv2.bitwise_or(active_mask, b_mask, dst=active_mask)

        # Restore mask = deleted region minus any active bubble regions
        restore_mask = cv2.bitwise_and(deleted_mask, cv2.bitwise_not(active_mask))
    else:
        restore_mask = deleted_mask

    # 3. Copy original pixels onto erased background
    restored_erased = erased_img.copy()
    restored_erased[restore_mask > 0] = original_img[restore_mask > 0]

    return restored_erased
