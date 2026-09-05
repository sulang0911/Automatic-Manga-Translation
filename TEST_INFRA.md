# Test Infrastructure & Verification Guide

## 1. Overview & Test Architecture

The `Automatic-Manga-Translation` verification suite is organized into distinct test tiers designed to balance rapid developer feedback, mathematical/geometric rigor, and end-to-end user experience.

### Test Tiers

| Tier | Directory | Scope & Description | Execution Time | Runner Command |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Unit Tests** | `tests/unit/` | Pure algorithms, geometric transforms, data models, bounding box clustering heuristics, QR immunity filtering, inpainting mask isolation, and typography calculations. Fully mocked external models (no GPU or network required). | ~15 - 20 seconds | `pytest tests/unit/ -v` |
| **Tier 2: E2E & GUI Tests** | `tests/e2e/` | PyQt6 offscreen UI workflow, canvas interactions, inspector panel synchronization, and end-to-end worker pipelines using synthetic manga pages. | ~30 - 60 seconds | `pytest tests/e2e/ -v` |
| **Tier 3: Challenge & Stress Tests** | `tests/challenge/` | Adversarial page layouts, extreme aspect ratios, overlapping bubbles, noisy OCR fragments, nested subfolder batches, and cache resilience. | ~30 - 45 seconds | `pytest tests/challenge/ -v` |

---

## 2. Feature Coverage Matrix

| Requirement | Feature Module | Target Source Files | Test Specification File | Test Cases / Scenarios |
| :--- | :--- | :--- | :--- | :--- |
| **R1: Rotated & Slanted Text** | OCR Polygon Angle Extraction | `app/core/ocr/base.py`<br>`desktop/core/ocr_engine.py`<br>`app/core/ocr/paddle_engine.py`<br>`app/core/ocr/easyocr_engine.py` | `tests/unit/test_ocr_angle_polygon.py` | - `calculate_polygon_angle` for 0°, +15°, -15°, and 45° quadrilaterals<br>- Angle normalization to [-90°, +90°]<br>- Small angle noise threshold (< 3.0° clamped to 0.0°) |
| **R1: Rotated Data Model** | TranslationBlock Polygon & Angle | `app/core/models.py` | `tests/unit/test_ocr_angle_polygon.py` | - `to_pixel_polygon` conversion with and without polygon<br>- `get_effective_angle` priority (`angle_override` vs `angle`)<br>- `from_pixel_box` and `to_dict` serialization round-trip |
| **R1 & R2: Angle & QR Merge Guard** | Adjacency Graph Gating | `app/core/ocr/base.py` (`can_merge_pair`) | `tests/unit/test_ocr_angle_polygon.py`<br>`tests/unit/test_qr_bubble_adaptive.py` | - Gating: pairs with \|Δθ\| <= 10° merge; \|Δθ\| > 10° rejected<br>- Merge span crossing or enclosing protected QR code rejected |
| **R1: Polygon Propagation** | Box Merging Polygon Math | `app/core/ocr/base.py` (`merge_adjacent_boxes`) | `tests/unit/test_ocr_angle_polygon.py` | - Merged box polygon generation via convex hull / bounding rect<br>- Component angle aggregation (median angle preservation) |
| **R2: QR & Barcode Detection** | `QRCodeFilter` & Geometry | `app/core/ocr/qr_filter.py` | `tests/unit/test_qr_bubble_adaptive.py` | - Detection on synthetic test image with checkerboard QR<br>- Detection and decode on real sample `exported_chapter/media_1788518641910.jpg`<br>- QR coordinate verification (46, 264, 129, 346) |
| **R2: OCR Noise Filtering** | Spurious Text Rejection | `app/core/ocr/qr_filter.py`<br>`desktop/core/ocr_engine.py` | `tests/unit/test_qr_bubble_adaptive.py` | - Discard text boxes falling inside QR bounding boxes<br>- Preserve valid dialogue blocks outside QR code |
| **R2: Inpainting QR Shield** | Inpainting Mask Immunity | `app/core/inpaint/opencv_engine.py`<br>`desktop/core/inpaint_engine.py` | `tests/unit/test_qr_bubble_adaptive.py` | - Inpainting mask zero-masking on QR areas (`inpaint_mask[qr > 0] = 0`)<br>- Exact 0-pixel change in QR area after inpaint |
| **R3: Adaptive Bubble Clustering** | Bubble-Aware Connected Graph | `app/core/ocr/base.py` | `tests/unit/test_qr_bubble_adaptive.py` | - Multi-line text with line spacing 1.8 ~ 2.2 * line_h in same bubble merged<br>- Separate dialogue bubbles with dark border rejected from merging<br>- Coordinate-only default preserving 100% backward compatibility with existing 170 unit tests |

---

## 3. Test Runners & Commands

### Prerequisites
- Python 3.10+
- Dependencies installed: `pytest`, `pytest-qt`, `numpy`, `opencv-python`, `Pillow`

### Running Baseline Unit Tests
```bash
pytest tests/unit/ -v
```

### Running Targeted Track Tests
```bash
# Test OCR polygon and angle calculations & gating
pytest tests/unit/test_ocr_angle_polygon.py -v

# Test QR code immunity filter & adaptive bubble clustering
pytest tests/unit/test_qr_bubble_adaptive.py -v
```

### Headless & CI/CD Execution
On headless machines or automated CI/CD runners, ensure the offscreen platform plugin is activated:
```bash
# Windows PowerShell
$env:QT_QPA_PLATFORM="offscreen"
pytest tests/unit/

# Linux / macOS Bash
export QT_QPA_PLATFORM=offscreen
pytest tests/unit/
```

---

## 4. Test Fixtures & Synthetic Generation

1. **Synthetic Slanted Lines**: Generated using exact affine rotation matrices and trigonometric vectors to eliminate external font rendering dependency.
2. **Synthetic QR Codes**: Generated using OpenCV high-contrast binary checkerboard patterns and finder squares, with optional `cv2.QRCodeEncoder` or real image validation on `exported_chapter/media_1788518641910.jpg`.
3. **Synthetic Speech Bubbles**: Generated using `cv2.ellipse` or `cv2.rectangle` with high interior luminance (255) and dark contour borders (0) on neutral background (200) to test connected-component morphology.
