# Test Infrastructure & Quality Assurance Specification

## 1. Testing Philosophy & Guiding Principles

The `Automatic-Manga-Translation` testing framework enforces rigorous, multi-tiered verification across all computer vision, optical character recognition, machine translation, image inpainting, and typography rendering modules. The framework adheres to four core principles:

1. **Zero-Mock Realism for Complex Visual Artifacts**:
   While unit tests utilize synthetic primitives for fast algorithmic verification, End-to-End (E2E) tests evaluate real-world manga pages from the Pixiv benchmark (`D:\baidu\download\baidu\pixiv\test`). Mocking is strictly forbidden in Tier 4 E2E benchmarks; all detection, clustering, inpainting, and typography algorithms must execute on actual image tensors.

2. **Strict Test Integrity & Anti-Cheating Invariants**:
   Tests must evaluate concrete geometrical, optical, and linguistic invariants (e.g. connected component isolation, ray-casting barrier detection, RGB pixel luminance standard deviations, Unicode script distributions). Facade tests, hardcoded true values, and tautological assertions are strictly prohibited.

3. **Progressive Testability & Isolation**:
   Every test suite is fully isolated and self-contained. Headless execution (`QT_QPA_PLATFORM=offscreen`) is guaranteed across both Windows and POSIX environments. Temporary files and output caches are managed using isolated fixtures (`tmp_path` or `.amt_cache` separation).

4. **Multi-Tier Verification Pyramid**:
   Tests are stratified into 4 progressive tiers, moving from fast unit feedback (< 20s) to comprehensive real-world multi-page visual evaluation.

---

## 2. 4-Tier Test Architecture

```
+-----------------------------------------------------------------------------------+
|               Tier 4: Real-World Manga Benchmark Suite (E2E)                      |
|   `tests/e2e/test_pixiv_10pages.py` (10 Pixiv Pages: Segmentation, OCR, Inpaint)  |
+-----------------------------------------------------------------------------------+
|               Tier 3: Adversarial & Stress Testing Suite                          |
|   `tests/challenge/` (Extreme Aspect Ratios, Overlaps, Corrupted Formats)         |
+-----------------------------------------------------------------------------------+
|               Tier 2: Synthetic E2E & Desktop Workflow Suites                     |
|   `tests/e2e/test_tier1_*.py` ~ `test_tier4_*.py` (Orchestrated by test_runner.py) |
|   GUI Canvas, Offscreen Workers, Async Cancellation, Model Dispatch               |
+-----------------------------------------------------------------------------------+
|               Tier 1: Fast Algorithmic Unit Tests (263 Tests)                     |
|   `tests/unit/` (Geometry, Angle Extraction, QR Immunity, Typography Math)        |
+-----------------------------------------------------------------------------------+
```

### Tier 1: Fast Algorithmic Unit Tests
- **Directory**: `tests/unit/`
- **Scope**: Mathematical transformations, polygon calculations, bounding box merging heuristics, connected-component bubble labeling, ray-casting boundary barrier logic, QR code filtering, stroke calculations, undo/redo history, and serialization models.
- **Execution Target**: Pure Python + OpenCV + NumPy (zero network, fast CPU).
- **Target Count**: 263 baseline unit tests.
- **Run Command**: `pytest tests/unit/ -q`

### Tier 2: Synthetic E2E & Desktop Pipeline Integration Tests
- **Directory**: `tests/e2e/` (`test_tier1_gui.py`, `test_tier1_ocr.py`, `test_tier1_inpaint.py`, `test_tier1_translation.py`, `test_tier1_typography.py`, `test_tier1_editing.py`, `test_tier1_export.py`, `test_tier1_async.py`, `test_tier1_error.py`, `test_tier2_boundary.py`, `test_tier3_combinations.py`, `test_tier4_scenarios.py`)
- **Scope**: Complete offscreen desktop application workflows, PyQt6 canvas selection, inspector property synchronization, async background worker thread management, and multi-step pipeline lifecycle.
- **Run Command**: `python tests/e2e/test_runner.py` or `pytest tests/e2e/test_tier*.py`

### Tier 3: Adversarial Challenge & Stress Tests
- **Directory**: `tests/challenge/`
- **Scope**: Stress testing against corrupt image files, huge resolutions (> 4K), degenerate zero-area boxes, extreme text densities, overlapping bubbles, and Unicode surrogate edge cases.
- **Run Command**: `pytest tests/challenge/ -v`

### Tier 4: Real-World 10-Image Pixiv Benchmark Suite
- **Test File**: `tests/e2e/test_pixiv_10pages.py`
- **Scope**: Comprehensive end-to-end evaluation on all 10 raw comic pages in `D:\baidu\download\baidu\pixiv\test` (3 color pages, 7 screentone pages).
- **Evaluation Dimensions**:
  1. **Dimension 1: Bubble Segmentation & Isolation**
     - Verify tight adjacent oval speech bubbles on `88061806_p002.jpg` are detected as 2 separate `TranslationBlock` instances, strictly NOT merged into 1.
     - Verify slanted external side annotations ("What a naughty maid!", "And you ruined his sister's dress...") are physically isolated from main speech bubbles.
  2. **Dimension 2: OCR Accuracy & Language Routing**
     - Verify English manga routes cleanly to EasyOCR, yielding 0 Japanese kana (hiragana/katakana) hallucinations.
     - Verify multi-line EasyOCR text is correctly sorted in 2D reading order (top-to-bottom, left-to-right) without word scrambling.
  3. **Dimension 3: Dark Box Inpainting & Typography**
     - Verify inverted dark/black boxes ("2 days and 1 masturbation later", "Friend A (19)", etc.) have clean background flat-fill with zero light-gray halos or smudges.
     - Verify typography renders high-contrast white text (`#FFFFFF`) with dark outline on dark backgrounds.
  4. **Dimension 4: Regression & Performance**
     - Verify invocation via `pytest tests/e2e/test_pixiv_10pages.py` and standalone `python tests/e2e/test_pixiv_10pages.py`.
     - Zero regression across all existing 263 unit tests.
- **Run Command**: `pytest tests/e2e/test_pixiv_10pages.py -v` or `python tests/e2e/test_pixiv_10pages.py`

---

## 3. Feature Inventory Coverage Matrix

| Feature ID | Feature Name | Source Module | Primary Test Tier | Test Specification File | Concrete Verification Invariants & Assertions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F1** | CTD Spatial & Boundary Line Splitting | `app/core/ocr/ctd_engine.py` | Tier 1, Tier 4 | `tests/unit/test_ocr_angle_polygon.py`<br>`tests/e2e/test_pixiv_10pages.py` | Split lines when vertical distance $y\_gap > 1.2 \times avg\_h$ or dark barrier pixels ($I < 90$) detected; verify separate blocks generated. |
| **F2** | Bubble Connected Component Refinement | `app/core/ocr/base.py` | Tier 1, Tier 4 | `tests/unit/test_qr_bubble_adaptive.py`<br>`tests/e2e/test_pixiv_10pages.py` | Morphological closing kernel $\le 7\times 7$ avoids merging adjacent bubbles; page background area filter prevents label zeroing. |
| **F3** | Ray-Casting Dark Boundary Barrier | `app/core/ocr/base.py` | Tier 1, Tier 4 | `tests/unit/test_qr_bubble_adaptive.py`<br>`tests/e2e/test_pixiv_10pages.py` | Sample pixel luminance along centroid-connecting segment in `can_merge_pair`; reject merge if dark barrier ($I < 90$) detected. |
| **F4** | Slanted Handwriting Boundary Protection | `app/core/ocr/base.py` | Tier 1, Tier 4 | `tests/unit/test_ocr_angle_polygon.py`<br>`tests/e2e/test_pixiv_10pages.py` | Angle difference $\| \Delta \theta \| \le 10^\circ$ gating; vector collinearity verification; slanted notes remain separate blocks. |
| **F5** | Fast English Pre-Detection & Hard Gating | `desktop/core/ocr_engine.py` | Tier 2, Tier 4 | `tests/e2e/test_tier1_ocr.py`<br>`tests/e2e/test_pixiv_10pages.py` | Identify Latin character presence; route to EasyOCR; hard-disable Manga-OCR; assert regex `[\u3040-\u30ff]` matches 0 characters. |
| **F6** | EasyOCR 2D Spatial Fragment Sorting | `desktop/core/ocr_engine.py`<br>`app/core/ocr/easyocr_engine.py` | Tier 1, Tier 4 | `tests/e2e/test_tier1_ocr.py`<br>`tests/e2e/test_pixiv_10pages.py` | Spatial line-clustering and $(y_{min}, x_{min})$ sorting prior to text joining; assert word order matches ground truth reading flow. |
| **F7** | Slanted Upright Crop Normalization | `app/core/ocr/easyocr_engine.py` | Tier 1, Tier 4 | `tests/unit/test_ocr_angle_polygon.py`<br>`tests/e2e/test_pixiv_10pages.py` | Affine rotation matrix $R(-\theta)$ applied to crops before feeding to EasyOCR; improves tilted word recognition accuracy. |
| **F8** | Translation Context & Punctuation Guard | `desktop/core/ocr_engine.py` | Tier 1, Tier 2 | `tests/e2e/test_tier1_translation.py`<br>`tests/e2e/test_pixiv_10pages.py` | Strip stray leading colons and orphan punctuation ("：衣服", "：天") caused by OCR line segmentation artifacts. |
| **F9** | Aspect-Ratio Onomatopoeia De-biasing | `desktop/core/ocr_engine.py` | Tier 1, Tier 4 | `tests/e2e/test_pixiv_10pages.py` | Narration banners with aspect ratio $> 4.0$ classified as `bubble` / `narration` when background is uniform (`is_uniform == True`). |
| **F10** | Inverted Dark Box Instant Flat-Fill | `app/core/inpaint/opencv_engine.py`<br>`desktop/core/inpaint_engine.py` | Tier 1, Tier 4 | `tests/unit/test_inpaint_mask.py`<br>`tests/e2e/test_pixiv_10pages.py` | Direct solid median fill for dark boxes (`block.bg_color == "#000000"` or $lum < 40$); bypass Gaussian alpha feather blending; assert $\mu(crop) < 5.0$, high-lum pixels $= 0$. |
| **F11** | Background-Color-Aware Typography | `app/core/typography/engine.py` | Tier 1, Tier 4 | `tests/unit/test_typography_settings.py`<br>`tests/e2e/test_pixiv_10pages.py` | Inspect stored `block.bg_color`; if dark ($lum < 60$), enforce white text `#FFFFFF` and contrasting stroke; assert text visibility. |
| **F12** | Automated 10-Image E2E Evaluation Suite | `tests/e2e/test_pixiv_10pages.py` | Tier 4 | `tests/e2e/test_pixiv_10pages.py` | Programmatic verification of all 10 real Pixiv images across the 4 quality dimensions. |
| **F13** | Zero-Regression Unit Test Guarantee | `tests/unit/` | Tier 1 | `pytest tests/unit/ -q` | 100% pass rate maintained across all 263 unit tests. |

---

## 4. Test Execution & CI/CD Guide

### Prerequisites
- Python 3.10+ (Current runtime: Python 3.12.6)
- Dependencies installed: `pytest`, `pytest-qt`, `numpy`, `opencv-python`, `Pillow`, `torch`, `easyocr`

### Environment Configuration
Headless offscreen operation is required for headless test runners:
```powershell
# Windows PowerShell
$env:QT_QPA_PLATFORM="offscreen"
```
```bash
# Linux / macOS
export QT_QPA_PLATFORM=offscreen
```

### Test Commands

1. **Run Full Baseline Unit Test Suite (Tier 1)**:
   ```bash
   pytest tests/unit/ -q
   ```
   *Expected Outcome*: 263 passed, 0 failed in ~45-50s.

2. **Run Synthetic 4-Tier E2E Suites (Tier 2)**:
   ```bash
   python tests/e2e/test_runner.py
   ```

3. **Run Real-World 10-Page Pixiv E2E Benchmark (Tier 4)**:
   ```bash
   # Via Pytest
   pytest tests/e2e/test_pixiv_10pages.py -v

   # Or via Standalone Executable Runner
   python tests/e2e/test_pixiv_10pages.py
   ```

4. **Run Full Combined Test Suite**:
   ```bash
   pytest tests/unit/ tests/e2e/test_pixiv_10pages.py -q
   ```
