# Test Readiness & Verification Certification

**Status**: READY  
**Date**: 2026-09-04  
**Target Project**: Automatic-Manga-Translation  
**Track**: E2E & Unit Testing Track  
**Total Unit Tests Passing**: 199 / 199 (100%)  
**Execution Time**: 17.34s  

---

## 1. Test Suite Summary

The unit test suite has been expanded with two dedicated test modules covering all aspects of tilted/rotated text polygon calculation, QR code isolation and immunity, and adaptive bubble clustering:

| Test Module | Test Focus | Tests | Status | Execution Time |
| :--- | :--- | :--- | :--- | :--- |
| `tests/unit/test_ocr_angle_polygon.py` | Polygon angle extraction (0°, ±15°, 45°), small angle deadband (<2.5°), `TranslationBlock` polygon & angle methods, `can_merge_pair` angle gating (\|Δθ\| <= 10°), OCR polygon & angle propagation in `merge_adjacent_boxes`. | 20 | PASSED | ~3.89s |
| `tests/unit/test_qr_bubble_adaptive.py` | OpenCV QR detection on synthetic images & real sample (`exported_chapter/media_1788518641910.jpg`), spurious OCR noise filtering, inpainting mask 0-pixel change guarantee in QR region, adaptive line spacing (1.8 ~ 2.2x line_h), bubble closure connected-component discrimination, coordinate-only default backward compatibility. | 9 | PASSED | ~4.49s |
| `tests/unit/` (Baseline Suite) | Reading order, models, typography, inpainting, recursive hierarchical export, GUI canvas, undo manager, etc. | 170 | PASSED | ~15.26s |
| **Total Combined Suite** | **Full Unit Test Track** | **199** | **ALL PASSED** | **17.34s** |

---

## 2. Requirement Coverage Verification

### Requirement R1: Rotated & Slanted Text Support
- **`calculate_polygon_angle`**:
  - Horizontal (0.0°): verified returning `0.0`.
  - Clockwise (+15.0°): verified returning `15.0 ± 0.5°`.
  - Counter-clockwise (-15.0°): verified returning `-15.0 ± 0.5°`.
  - 45° tilted quadrilateral: verified returning `45.0 ± 0.5°`.
  - Deadband noise threshold: verified angles with \|deg\| < 2.5° clamp to `0.0`.
  - General polygon fallback: verified contour minAreaRect calculation.
- **`TranslationBlock` Methods**:
  - `get_effective_angle()`: verified precedence of `angle_override` over `angle`.
  - `to_pixel_polygon()`: verified conversion for explicit polygon, horizontal synthesized box, and rotated synthesized box.
  - Serialization: verified round-trip preservation of `polygon` and `angle` via `to_dict` and `from_dict`.
- **Angle Gating in `can_merge_pair`**:
  - Pairs with \|Δθ\| <= 10° merge cleanly (`True`).
  - Pairs with \|Δθ\| > 10° strictly rejected (`False`).
  - Boundary condition: \|Δθ\| = 10.0° allowed; \|Δθ\| = 10.5° rejected.
  - Missing/None angle: 100% backward compatible.
- **Polygon & Angle Propagation in `merge_adjacent_boxes`**:
  - Single box preserves polygon and angle.
  - Multi-line merge computes median angle and synthesizes oriented bounding polygon covering all vertices.
  - Incompatible angle boxes remain separated.

### Requirement R2: QR Code & Barcode Immunity
- **OpenCV QR Detection**:
  - Synthetic QR image: generated and verified with `cv2.QRCodeDetector()` decoding exact URL and bounding box.
  - Real sample `exported_chapter/media_1788518641910.jpg`: verified identifying QR code at `(46, 264, 129, 346)` and decoding `https://nimbletail.com/`.
- **Spurious OCR Noise Filtering**:
  - Verified 100% rejection of false-positive OCR character noise falling inside the QR code region.
  - Verified preservation of legitimate speech bubbles outside the QR code region.
  - Verified partial overlap (>60% area inside QR) rejection.
- **Inpainting Protection**:
  - Verified inpainting mask zero-masking on QR region (`inpaint_mask[qr > 0] = 0`).
  - Verified exact 0-pixel modification (`np.array_equal` True) on the QR area after inpainting.
  - Verified QR code remains fully decodable via OpenCV after inpainting.
  - Verified on real sample `media_1788518641910.jpg`.

### Requirement R3: Adaptive Bubble & Line Grouping
- **Coordinate-Only Backward Compatibility**:
  - Under default coordinate-only parameters (`v_thresh_ratio=0.025`), two boxes with line gap = 1.0x line height (> 0.70 * line_h) remain separate.
  - Guarantees 100% regression safety for existing 170 unit tests.
- **Adaptive Line Spacing ($1.8 \sim 2.2 \times line\_h$)**:
  - When adaptive threshold is active (`v_thresh_ratio=0.075`), multi-line text with line spacing 1.8 ~ 2.0x line height successfully merges into a single `TranslationBlock`.
- **Bubble Closure Connected-Component Awareness**:
  - Verified morphological connected-component labeling (`cv2.threshold` + `morphologyEx` + `connectedComponents`).
  - Text lines sharing the same continuous speech bubble closure have matching component labels (`lbl1 == lbl2 > 0`).
  - Text lines across independent bordered speech bubbles have distinct labels (`lbl1 != lbl2`), preventing false-positive cross-bubble merging.

---

## 3. How to Execute the Test Suite

```bash
# Run the newly added OCR angle and polygon tests:
pytest tests/unit/test_ocr_angle_polygon.py -v

# Run the newly added QR immunity and adaptive bubble clustering tests:
pytest tests/unit/test_qr_bubble_adaptive.py -v

# Run the complete unit test suite (199 tests):
pytest tests/unit/ -q
```

---

## 4. Verification Evidence

```
============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\baidu\download\test\Automatic-Manga-Translation
plugins: anyio-4.15.0
collected 199 items

199 passed in 17.34s
============================ 199 passed in 17.34s =============================
```

**Certification**: All unit tests compile, execute cleanly, and pass with zero failures and zero regressions.
