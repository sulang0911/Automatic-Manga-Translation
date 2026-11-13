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
    Extracts polygon point arrays and pixel bounding boxes associated with a block,
    including its current polygon/box, pre-drag original geometry (_orig_box, _orig_polygon),
    and merged constituent boxes (_merged_boxes). Polygons are not inflated into AABBs.
    """
    polys: List[np.ndarray] = []
    boxes: List[Tuple[int, int, int, int]] = []

    def _norm_to_px_box(
        xmin_n: float, ymin_n: float, xmax_n: float, ymax_n: float
    ) -> Optional[Tuple[int, int, int, int]]:
        x1 = max(0, min(img_w, int(round(float(xmin_n) * img_w / 100.0))))
        y1 = max(0, min(img_h, int(round(float(ymin_n) * img_h / 100.0))))
        x2 = max(0, min(img_w, int(round(float(xmax_n) * img_w / 100.0))))
        y2 = max(0, min(img_h, int(round(float(ymax_n) * img_h / 100.0))))
        return (x1, y1, x2, y2) if (x2 > x1 and y2 > y1) else None

    def _add_norm_box(xmin_n: float, ymin_n: float, xmax_n: float, ymax_n: float):
        b = _norm_to_px_box(xmin_n, ymin_n, xmax_n, ymax_n)
        if b is not None:
            boxes.append(b)

    def _parse_poly(raw_p: Any) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        if raw_p and len(raw_p) >= 3:
            try:
                pts = np.array([[int(round(p[0])), int(round(p[1]))] for p in raw_p], dtype=np.int32)
                px1 = max(0, min(img_w, int(np.min(pts[:, 0]))))
                py1 = max(0, min(img_h, int(np.min(pts[:, 1]))))
                px2 = max(0, min(img_w, int(np.max(pts[:, 0]))))
                py2 = max(0, min(img_h, int(np.max(pts[:, 1]))))
                if px2 > px1 and py2 > py1:
                    return pts, (px1, py1, px2, py2)
            except Exception:
                pass
        return None, None

    def _add_shape(
        raw_p: Any,
        norm_box_tuple: Optional[Tuple[float, float, float, float]],
        synth_tb: Optional[TranslationBlock] = None,
    ):
        pts, pb = _parse_poly(raw_p)
        from_synth = False
        if pts is None and synth_tb is not None:
            pts, pb = _parse_poly(synth_tb.to_pixel_polygon(img_w, img_h))
            from_synth = pts is not None
        nbox = _norm_to_px_box(*norm_box_tuple) if norm_box_tuple is not None else None
        if pts is not None:
            polys.append(pts)
            # Only also include norm_box if an explicit polygon disagrees significantly (shifted box)
            if not from_synth and nbox is not None and pb is not None:
                if any(abs(nbox[i] - pb[i]) > 4 for i in range(4)):
                    boxes.append(nbox)
        elif nbox is not None:
            boxes.append(nbox)

    is_tb = isinstance(block, TranslationBlock)
    get_attr = (lambda k, d=None: getattr(block, k, d)) if is_tb else block.get

    if is_tb:
        raw_p = block.polygon if (block.polygon and len(block.polygon) >= 3) else None
        synth = block if (raw_p is None and abs(block.get_effective_angle()) >= 2.5) else None
        _add_shape(raw_p, (block.xmin, block.ymin, block.xmax, block.ymax), synth_tb=synth)
    elif isinstance(block, dict):
        raw_poly = block.get("polygon")
        synth = None
        if not (raw_poly and len(raw_poly) >= 3):
            try:
                tb = TranslationBlock.from_dict(block)
                if abs(tb.get_effective_angle()) >= 2.5:
                    synth = tb
            except Exception:
                pass
        _add_shape(
            raw_poly,
            (block.get("xmin", 0.0), block.get("ymin", 0.0), block.get("xmax", 0.0), block.get("ymax", 0.0)),
            synth_tb=synth,
        )

    orig_box = get_attr("_orig_box")
    orig_poly = get_attr("_orig_polygon")
    ob_tuple = tuple(orig_box) if (orig_box and len(orig_box) == 4) else None
    if orig_poly or ob_tuple:
        _add_shape(orig_poly, ob_tuple)

    merged_boxes = get_attr("_merged_boxes")
    if merged_boxes:
        for mb in merged_boxes:
            if isinstance(mb, dict):
                _add_shape(
                    mb.get("polygon"),
                    (mb.get("xmin", 0.0), mb.get("ymin", 0.0), mb.get("xmax", 0.0), mb.get("ymax", 0.0)),
                )
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


def _calc_tight_rem_pad(img_w: int, img_h: int, padding: int) -> int:
    """Compact outer halo margin used for active neighbor protection."""
    return max(padding, 4) + max(4, int(round(max(img_w, img_h) * 0.003)) * 2 + 2)


def _get_core_and_expanded_inpaint_mask(
    block: Union[Dict[str, Any], TranslationBlock],
    img_w: int,
    img_h: int,
    padding: int = 4,
    tight: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes both the tight core mask of a block and its expanded inpainting footprint.
    When `tight=False` (for the deleted block), uses full unboxed/stroked/feather margins
    so halos restore cleanly. When `tight=True` (for remaining neighbor blocks), uses a
    compact outer feather margin matching actual inpainter dilation + feather bleed.
    """
    core = get_block_pixel_mask(block, img_w, img_h, padding=0)
    exp = np.zeros((img_h, img_w), dtype=np.uint8)
    if not np.any(core):
        return core, exp

    if tight:
        pad = _calc_tight_rem_pad(img_w, img_h, padding)
        ys, xs = np.nonzero(core)
        x1 = max(0, int(xs.min()) - pad)
        y1 = max(0, int(ys.min()) - pad)
        x2 = min(img_w, int(xs.max()) + pad + 1)
        y2 = min(img_h, int(ys.max()) + pad + 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        exp[y1:y2, x1:x2] = cv2.dilate(core[y1:y2, x1:x2], k)
        return core, exp

    polys, boxes = _extract_block_geometries(block, img_w, img_h)
    if not polys and not boxes:
        ys, xs = np.nonzero(core)
        boxes = [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))]

    feather_margin = max(18, int(round(max(img_w, img_h) * 0.003)) * 3 + 12)

    def _calc_pads(bw: int, bh: int) -> Tuple[int, int]:
        if bw >= bh:
            return (
                max(padding, 10, int(round(bw * 0.05)), int(round(bh * 0.30))) + feather_margin,
                max(padding, 8, int(round(bh * 0.35))) + feather_margin,
            )
        return (
            max(padding, 8, int(round(bw * 0.35))) + feather_margin,
            max(padding, 10, int(round(bh * 0.05)), int(round(bw * 0.30))) + feather_margin,
        )

    for (x1, y1, x2, y2) in boxes:
        pad_x, pad_y = _calc_pads(max(1, x2 - x1), max(1, y2 - y1))
        ex1 = max(0, x1 - pad_x)
        ey1 = max(0, y1 - pad_y)
        ex2 = min(img_w, x2 + pad_x)
        ey2 = min(img_h, y2 + pad_y)
        if ex2 > ex1 and ey2 > ey1:
            cv2.rectangle(exp, (ex1, ey1), (ex2, ey2), 255, -1)

    for poly_pts in polys:
        rect = cv2.minAreaRect(poly_pts.astype(np.float32))
        rw, rh = max(1.0, float(rect[1][0])), max(1.0, float(rect[1][1]))
        pad_major, pad_minor = _calc_pads(int(round(max(rw, rh))), int(round(min(rw, rh))))
        max_pad = max(pad_major, pad_minor)
        px1 = max(0, int(np.min(poly_pts[:, 0])) - max_pad)
        py1 = max(0, int(np.min(poly_pts[:, 1])) - max_pad)
        px2 = min(img_w, int(np.max(poly_pts[:, 0])) + max_pad + 1)
        py2 = min(img_h, int(np.max(poly_pts[:, 1])) + max_pad + 1)
        if px2 <= px1 or py2 <= py1:
            continue
        local = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
        cv2.fillPoly(local, [poly_pts - np.array([px1, py1], dtype=np.int32)], 255)
        dx_box = int(np.max(poly_pts[:, 0])) - int(np.min(poly_pts[:, 0]))
        dy_box = int(np.max(poly_pts[:, 1])) - int(np.min(poly_pts[:, 1]))
        if abs(rect[2]) < 3 or abs(abs(rect[2]) - 90) < 3:
            px = pad_major if dx_box >= dy_box else pad_minor
            py = pad_minor if dx_box >= dy_box else pad_major
            k = np.ones((2 * py + 1, 2 * px + 1), dtype=np.uint8)
        else:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max_pad + 1, 2 * max_pad + 1))
        cv2.bitwise_or(exp[py1:py2, px1:px2], cv2.dilate(local, k), dst=exp[py1:py2, px1:px2])

    return core, exp


def _match_channels(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Converts src channel layout to match ref when spatial dimensions match."""
    src_c = 1 if src.ndim == 2 else src.shape[2]
    ref_c = 1 if ref.ndim == 2 else ref.shape[2]
    if src_c == ref_c:
        return src
    if ref_c == 3:
        if src_c == 1:
            return cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        if src_c == 4:
            return cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
    elif ref_c == 4:
        if src_c == 1:
            return cv2.cvtColor(src, cv2.COLOR_GRAY2BGRA)
        if src_c == 3:
            return cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)
    elif ref_c == 1:
        if src_c == 3:
            return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        if src_c == 4:
            return cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
    return src


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
    original_img = _match_channels(original_img, erased_img)

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
    del_core, del_exp = _get_core_and_expanded_inpaint_mask(
        deleted_block, w_img, h_img, padding=padding, tight=False
    )
    if not np.any(del_exp):
        return erased_img.copy()

    if remaining_blocks:
        rem_core = np.zeros((h_img, w_img), dtype=np.uint8)
        rem_exp = np.zeros((h_img, w_img), dtype=np.uint8)
        for b in remaining_blocks:
            rc, re = _get_core_and_expanded_inpaint_mask(b, w_img, h_img, padding=padding, tight=True)
            if np.any(rc):
                cv2.bitwise_or(rem_core, rc, dst=rem_core)
            if np.any(re):
                cv2.bitwise_or(rem_exp, re, dst=rem_exp)

        if np.any(rem_core):
            kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            rem_core_dilated = cv2.dilate(rem_core, kernel_3)
            overlap = cv2.bitwise_and(del_exp, rem_exp)
            if np.any(overlap):
                rem_pad = _calc_tight_rem_pad(w_img, h_img, padding)
                # Only allow remaining blocks' outer halo to block restoration where they actually erased pixels
                ys_r, xs_r = np.nonzero(rem_core_dilated)
                rx1, ry1 = max(0, int(xs_r.min()) - rem_pad), max(0, int(ys_r.min()) - rem_pad)
                rx2, ry2 = min(w_img, int(xs_r.max()) + rem_pad + 1), min(h_img, int(ys_r.max()) + rem_pad + 1)
                diff_roi = cv2.absdiff(erased_img[ry1:ry2, rx1:rx2], original_img[ry1:ry2, rx1:rx2])
                diff_max = diff_roi.max(axis=2) if diff_roi.ndim >= 3 else diff_roi
                active_roi = np.where((rem_core_dilated[ry1:ry2, rx1:rx2] > 0) & (diff_max > 6), np.uint8(255), np.uint8(0))
                if np.any(active_roi):
                    k_act = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rem_pad + 1, 2 * rem_pad + 1))
                    active_halo = np.zeros((h_img, w_img), dtype=np.uint8)
                    active_halo[ry1:ry2, rx1:rx2] = cv2.dilate(active_roi, k_act)
                    rem_exp_eff = cv2.bitwise_and(rem_exp, active_halo)
                    if np.any(cv2.bitwise_and(del_exp, rem_exp_eff)):
                        dist_del = cv2.distanceTransform(cv2.bitwise_not(del_core), cv2.DIST_L2, 3)
                        dist_rem = cv2.distanceTransform(cv2.bitwise_not(rem_core), cv2.DIST_L2, 3)
                        halo_restore = (del_exp > 0) & (rem_core_dilated == 0) & ((rem_exp_eff == 0) | (dist_del <= dist_rem))
                    else:
                        halo_restore = (del_exp > 0) & (rem_core_dilated == 0)
                else:
                    halo_restore = (del_exp > 0) & (rem_core_dilated == 0)
            else:
                halo_restore = (del_exp > 0) & (rem_core_dilated == 0)
            # Never let neighbor halo/dilation block pixels strictly inside deleted_block's own core
            restore_bool = ((del_core > 0) & (rem_core == 0)) | halo_restore
        else:
            restore_bool = (del_exp > 0)
    else:
        restore_bool = (del_exp > 0)

    restored_erased = erased_img.copy()
    restored_erased[restore_bool] = original_img[restore_bool]

    return restored_erased
