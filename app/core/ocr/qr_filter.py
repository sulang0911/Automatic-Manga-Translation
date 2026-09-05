"""
app/core/ocr/qr_filter.py
Lightweight 2D QR code and 1D barcode detection, spurious OCR box rejection,
and immunity shield masking.
Uses cv2.QRCodeDetector, ArUco QR detection, cv2.barcode, and geometric contour heuristics.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import cv2


@dataclass
class QRRegion:
    """Represents a protected non-text visual asset (QR code or barcode)."""
    kind: str = "qrcode"  # "qrcode" | "barcode"
    polygon: Optional[List[List[int]]] = None  # Shape (4, 2) integer pixel vertices
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (xmin, ymin, xmax, ymax) in pixel coords
    bbox_normalized: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # (xmin, ymin, xmax, ymax) in [0, 100]
    text: str = ""
    decoded_text: str = ""

    def __post_init__(self):
        if not self.decoded_text and self.text:
            self.decoded_text = self.text
        elif not self.text and self.decoded_text:
            self.text = self.decoded_text

    def __getitem__(self, item):
        return getattr(self, item)


# Alias for backward compatibility with design documents
ProtectedRegion = QRRegion


class QRCodeFilter:
    """
    Detects QR codes and barcodes across input manga pages and provides
    immunity masks and geometric exclusion utilities.
    """

    def __init__(self, padding_pixels: int = 4):
        self.padding_pixels = max(0, int(padding_pixels))
        try:
            if hasattr(cv2, "setLogLevel"):
                cv2.setLogLevel(0)
        except Exception:
            pass
        self._qr_detector = cv2.QRCodeDetector()
        self._aruco_detector = None
        if hasattr(cv2, "QRCodeDetectorAruco"):
            try:
                self._aruco_detector = cv2.QRCodeDetectorAruco()
            except Exception:
                self._aruco_detector = None
        self._barcode_detector = None
        if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
            try:
                self._barcode_detector = cv2.barcode.BarcodeDetector()
            except Exception:
                self._barcode_detector = None

    def detect_regions(self, image: np.ndarray) -> List[QRRegion]:
        """
        Detects 2D QR codes and barcodes in an image.
        Returns a list of QRRegion objects containing pixel bbox, polygon, and decoded text.
        """
        if image is None or image.size == 0:
            return []

        h_img, w_img = image.shape[:2]
        regions: List[QRRegion] = []
        detected_boxes: List[Tuple[int, int, int, int]] = []

        # 1. Primary: cv2.QRCodeDetector
        try:
            ok, decoded_info, points, _ = self._qr_detector.detectAndDecodeMulti(image)
            if ok and points is not None and len(points) > 0:
                for idx, pts in enumerate(points):
                    if pts is not None and len(pts) >= 4:
                        dec = decoded_info[idx] if idx < len(decoded_info) else ""
                        reg = self._build_region("qrcode", pts, w_img, h_img, dec)
                        if reg:
                            regions.append(reg)
                            detected_boxes.append(reg.bbox)
            else:
                # Single fallback
                val, pts, _ = self._qr_detector.detectAndDecode(image)
                if pts is not None and len(pts) > 0:
                    pts_arr = pts[0] if pts.ndim == 3 else pts
                    reg = self._build_region("qrcode", pts_arr, w_img, h_img, val)
                    if reg:
                        regions.append(reg)
                        detected_boxes.append(reg.bbox)
        except Exception:
            pass

        # 2. ArUco-based QR detector (robust against perspective/rotations)
        if not regions and self._aruco_detector is not None:
            try:
                ok, decoded_info, points, _ = self._aruco_detector.detectAndDecodeMulti(image)
                if ok and points is not None and len(points) > 0:
                    for idx, pts in enumerate(points):
                        dec = decoded_info[idx] if idx < len(decoded_info) else ""
                        reg = self._build_region("qrcode", pts, w_img, h_img, dec)
                        if reg:
                            regions.append(reg)
                            detected_boxes.append(reg.bbox)
            except Exception:
                pass

        # 3. 1D Barcode detector
        if self._barcode_detector is not None:
            try:
                ok, decoded_info, _, points = self._barcode_detector.detectAndDecodeMulti(image)
                if ok and points is not None and len(points) > 0:
                    for idx, pts in enumerate(points):
                        dec = decoded_info[idx] if idx < len(decoded_info) else ""
                        reg = self._build_region("barcode", pts, w_img, h_img, dec)
                        if reg and not any(self._bbox_overlap(reg.bbox, b) > 0.5 for b in detected_boxes):
                            regions.append(reg)
                            detected_boxes.append(reg.bbox)
            except Exception:
                pass

        # 4. Auxiliary Contour & High-Frequency Texture Heuristic (Fallback for degraded/stylized QR)
        if not regions:
            heuristic_regs = self._detect_qr_by_contour_heuristic(image)
            regions.extend(heuristic_regs)

        return regions

    def _build_region(
        self,
        kind: str,
        pts: np.ndarray,
        w_img: int,
        h_img: int,
        text: str
    ) -> Optional[QRRegion]:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 4:
            return None
        xmin = int(np.floor(np.min(pts[:, 0]))) - self.padding_pixels
        ymin = int(np.floor(np.min(pts[:, 1]))) - self.padding_pixels
        xmax = int(np.ceil(np.max(pts[:, 0]))) + self.padding_pixels
        ymax = int(np.ceil(np.max(pts[:, 1]))) + self.padding_pixels

        xmin = max(0, min(xmin, w_img - 1))
        ymin = max(0, min(ymin, h_img - 1))
        xmax = max(xmin + 1, min(xmax, w_img))
        ymax = max(ymin + 1, min(ymax, h_img))

        pad_pts = [
            [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]
        ]

        norm_bbox = (
            round((xmin / w_img) * 100.0, 2),
            round((ymin / h_img) * 100.0, 2),
            round((xmax / w_img) * 100.0, 2),
            round((ymax / h_img) * 100.0, 2),
        )
        return QRRegion(
            kind=kind,
            polygon=pad_pts,
            bbox=(xmin, ymin, xmax, ymax),
            bbox_normalized=norm_bbox,
            text=str(text or ""),
            decoded_text=str(text or "")
        )

    def _detect_qr_by_contour_heuristic(self, image: np.ndarray) -> List[QRRegion]:
        """Detects square nested finder patterns and high-frequency checkerboard grid."""
        h_img, w_img = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or len(contours) == 0:
            return []

        # Find nested concentric squares (outer, middle, inner: 3 levels of nesting)
        finder_boxes = []
        for i, c in enumerate(contours):
            child_idx = hierarchy[0][i][2]
            if child_idx != -1:
                grandchild_idx = hierarchy[0][child_idx][2]
                if grandchild_idx != -1:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect = w / float(max(1, h))
                        if 0.8 <= aspect <= 1.25 and 10 <= w <= 150:
                            finder_boxes.append((x, y, w, h))

        if len(finder_boxes) >= 3:
            pts = np.array([[fb[0], fb[1]] for fb in finder_boxes] +
                           [[fb[0] + fb[2], fb[1] + fb[3]] for fb in finder_boxes])
            xmin, ymin = pts.min(axis=0)
            xmax, ymax = pts.max(axis=0)
            w = xmax - xmin
            h = ymax - ymin
            aspect = w / float(max(1, h))
            if 0.75 <= aspect <= 1.30 and 40 <= w <= 500:
                crop_thresh = thresh[ymin:ymax, xmin:xmax]
                transitions = np.sum(np.diff(crop_thresh.astype(int), axis=1) != 0)
                if transitions > 25:
                    reg = self._build_region(
                        "qrcode",
                        np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]),
                        w_img,
                        h_img,
                        ""
                    )
                    if reg:
                        return [reg]
        return []

    def filter_spurious_ocr_boxes(
        self,
        boxes: List[Dict[str, Any]],
        qr_regions: List[QRRegion],
        w_img: Optional[int] = None,
        h_img: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Filters out spurious OCR text boxes whose center or >60% area falls inside detected QR regions.
        Handles both pixel coordinates and normalized [0, 100] coordinates.
        """
        if not qr_regions or not boxes:
            return boxes

        filtered: List[Dict[str, Any]] = []
        for b in boxes:
            bx1 = min(b.get("xmin", 0), b.get("xmax", 0))
            by1 = min(b.get("ymin", 0), b.get("ymax", 0))
            bx2 = max(b.get("xmin", 0), b.get("xmax", 0))
            by2 = max(b.get("ymin", 0), b.get("ymax", 0))

            is_normalized = (bx2 <= 100.0 and by2 <= 100.0)

            is_inside_qr = False
            for reg in qr_regions:
                if is_normalized and w_img is None:
                    # Compare in normalized coordinates
                    qx1, qy1, qx2, qy2 = reg.bbox_normalized
                else:
                    # Compare in pixel coordinates
                    qx1, qy1, qx2, qy2 = reg.bbox
                    if is_normalized and w_img is not None and h_img is not None:
                        bx1_px = (bx1 / 100.0) * w_img
                        by1_px = (by1 / 100.0) * h_img
                        bx2_px = (bx2 / 100.0) * w_img
                        by2_px = (by2 / 100.0) * h_img
                    else:
                        bx1_px, by1_px, bx2_px, by2_px = bx1, by1, bx2, by2

                    cx = (bx1_px + bx2_px) / 2.0
                    cy = (by1_px + by2_px) / 2.0
                    if qx1 <= cx <= qx2 and qy1 <= cy <= qy2:
                        is_inside_qr = True
                        break

                    b_area = max(1.0, (bx2_px - bx1_px) * (by2_px - by1_px))
                    ix1 = max(bx1_px, qx1)
                    iy1 = max(by1_px, qy1)
                    ix2 = min(bx2_px, qx2)
                    iy2 = min(by2_px, qy2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        if (inter_area / b_area) >= 0.60:
                            is_inside_qr = True
                            break
                    continue

                # Normalized comparison branch
                cx = (bx1 + bx2) / 2.0
                cy = (by1 + by2) / 2.0
                if qx1 <= cx <= qx2 and qy1 <= cy <= qy2:
                    is_inside_qr = True
                    break
                b_area = max(0.0001, (bx2 - bx1) * (by2 - by1))
                ix1 = max(bx1, qx1)
                iy1 = max(by1, qy1)
                ix2 = min(bx2, qx2)
                iy2 = min(by2, qy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    if (inter_area / b_area) >= 0.60:
                        is_inside_qr = True
                        break

            if not is_inside_qr:
                filtered.append(b)

        return filtered

    def get_protection_mask(
        self,
        image_shape: Union[Tuple[int, int], Tuple[int, int, int], np.ndarray],
        qr_regions: Optional[List[QRRegion]] = None
    ) -> np.ndarray:
        """
        Returns a binary uint8 mask (255 for protected QR/barcode pixels, 0 elsewhere).
        Accepts either an image_shape tuple (h, w) or an image np.ndarray.
        """
        if isinstance(image_shape, np.ndarray):
            h, w = image_shape.shape[:2]
            if qr_regions is None:
                qr_regions = self.detect_regions(image_shape)
        else:
            h, w = int(image_shape[0]), int(image_shape[1])
            if qr_regions is None:
                qr_regions = []

        mask = np.zeros((h, w), dtype=np.uint8)
        if not qr_regions:
            return mask

        for reg in qr_regions:
            if reg.polygon is not None and len(reg.polygon) >= 3:
                pts = np.array(reg.polygon, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
            else:
                x1, y1, x2, y2 = reg.bbox
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                mask[y1:y2, x1:x2] = 255

        return mask

    @staticmethod
    def _bbox_overlap(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        inter_w = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        inter_h = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        inter = inter_w * inter_h
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        return inter / float(max(1, a1))
