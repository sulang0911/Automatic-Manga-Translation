# Project: Automatic-Manga-Translation Architecture Refactor & Upgrade

## Architecture
- **OCR Engine Pipeline (app/core/ocr/, desktop/core/ocr_engine.py)**:
  - Raw detection polygons (pts / polygon) extracted and preserved.
  - Line angle computed via baseline vector arctan2 / cv2.minAreaRect.
  - Box clustering in app/core/ocr/base.py with angle gating, bubble contour awareness, and adaptive spacing.
  - Spurious OCR fragment filtering via QRCodeFilter.
- **Data Models (app/core/models.py)**:
  - TranslationBlock with polygon, angle, angle_override, to_pixel_polygon(), get_effective_angle().
- **Inpainting Pipeline (app/core/inpaint/, desktop/core/inpaint_engine.py)**:
  - cv2.fillPoly + adaptive dilation for oriented polygons.
  - QR Code immunity shield (inpaint_mask[qr_mask > 0] = 0), zero pixel modification on QR codes.
- **Typography Engine (app/core/typography/engine.py)**:
  - Tilted text line-wrapping along oriented dimensions (L, H).
  - Bicubic rotated rendering centered at block centroid (cx, cy).
- **PyQt6 Workbench GUI (app/ui/canvas/, app/ui/inspector/, app/ui/main_window.py)**:
  - Live interactive angle adjustment, canvas item rotation, and override synchronization.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | OCR Angle & Polygon Extraction | Preserve 4-pt polygon, compute orientation angle in Paddle/EasyOCR/Desktop OCR | M1 | R1 Spec |
| 2 | Model Polygon & Angle Methods | to_pixel_polygon, get_effective_angle on TranslationBlock | M1 | R1 Spec |
| 3 | Inpainting Oriented Polygon Mask | cv2.fillPoly + adaptive dilation instead of AABB | M2 | R1 Spec |
| 4 | Typography Rotated Rendering | Oriented (L, H) text fitting, bicubic rotation around center | M2 | R1 Spec |
| 5 | PyQt6 Canvas & Inspector Angle Control | Dynamic angle slider, canvas item live rotation, override fix | M2 | R1 Spec |
| 6 | QR Code & Barcode Detector Filter | QRCodeFilter with OpenCV QR/ArUco/Barcode + geometric heuristics | M3 | R2 Spec |
| 7 | Inpaint & Typography QR Protection | Zero-masking QR regions during inpainting, prevent text overlay | M3 | R2 Spec |
| 8 | Spurious OCR Noise Filtering | Discard OCR text blocks falling inside detected QR codes | M3 | R2 Spec |
| 9 | Adaptive Line Spacing Grouping | Dynamic tolerance 1.8~2.5x line_h for well-aligned lines | M4 | R3 Spec |
| 10 | Bubble-Aware Connected Clustering | Group lines sharing connected bubble closure; preserve 100% backward compat | M4 | R3 Spec |
| 11 | Reading Flow & Merge Guards | Manga RTL / Western TTB preservation; angle & color & QR barrier guards | M4 | R3 Spec |
| 12 | Automated Unit & Visual Test Suite | Tests for slanted text, QR immunity, large bubble clustering in tests/unit/ | M5 | R4 Spec |
| 13 | Full Regression & GUI Verification | Verify 170+ existing tests pass, verify PyQt6 GUI launches without error | M5 | R4 Spec |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | OCR Angle & Polygon Pipeline | app/core/models.py, app/core/ocr/base.py, paddle_engine.py, easyocr_engine.py, desktop/core/ocr_engine.py | none | DONE |
| M2 | Inpainting & Rotated Typography & GUI | app/core/inpaint/, desktop/core/inpaint_engine.py, app/core/typography/engine.py, app/ui/canvas/, app/ui/inspector/ | M1 | DONE |
| M3 | QR Code Immunity Filter & Shields | app/core/ocr/qr_filter.py, pipeline integration in OCR, Inpainting, Typography | M1 | DONE |
| M4 | Adaptive Bubble Clustering & Reading Flow | app/core/ocr/base.py, connected component awareness, reading order | M1, M3 | DONE |
| M5 | Test Suite, Regression & Acceptance | tests/unit/test_ocr_angle_polygon.py, tests/unit/test_qr_bubble_adaptive.py, full pytest suite, GUI verification | M1, M2, M3, M4 | DONE |

## Interface Contracts
### TranslationBlock
- to_pixel_polygon(img_w: int, img_h: int) -> Optional[List[List[int]]]
- get_effective_angle() -> float: Returns angle_override if set else angle.

### QRCodeFilter
- detect_regions(image: np.ndarray) -> List[QRRegion]
- filter_spurious_ocr_boxes(boxes: List[Dict], qr_regions: List[QRRegion]) -> List[Dict]
- get_protection_mask(image_shape: Tuple[int, int], qr_regions: List[QRRegion]) -> np.ndarray

### app/core/ocr/base.py
- calculate_polygon_angle(pts: np.ndarray) -> float
- can_merge_pair(b1, b2, img_w, img_h, v_thresh_ratio, h_thresh_ratio, ...): Includes angle mismatch guard, QR barrier guard, color distance guard.
- merge_adjacent_boxes(raw_boxes, img_w, img_h, ..., image=None, adaptive_spacing=False): Bubble-aware connected clustering.

## Code Layout
- app/core/models.py: Core data structures
- app/core/ocr/: OCR engines, reading order, QR filter
- app/core/inpaint/: Inpainting engines
- app/core/typography/: Typography and rotated rendering
- app/ui/: PyQt6 UI workbench
- desktop/core/: Desktop standalone worker engines
- tests/unit/: Unit tests
