# Project: Automatic-Manga-Translation End-to-End Quality Optimization

## Architecture
- **Detection & Bubble Segmentation Layer**:
  - `app/core/ocr/ctd_engine.py`: Comic Text Detector (DBNet + textline clustering).
  - `app/core/ocr/base.py`: Connected components (`compute_bubble_labels`), pairwise line merging (`can_merge_pair`, `merge_adjacent_boxes`), ray-casting dark boundary barriers.
- **OCR & Language Perception Layer**:
  - `desktop/core/ocr_engine.py`: Integrated desktop OCR pipeline with language routing, CRAFT/EasyOCR detection and recognition.
  - `app/core/ocr/easyocr_engine.py`: Direct EasyOCR integration with upright polygon rotation and 2D spatial fragment sorting.
  - `app/core/ocr/mangaocr_engine.py`: Japanese-only OCR engine (gated behind strict language detection).
  - `app/core/pipeline.py` & `desktop/workers/pipeline_worker.py`: Language auto-detection and execution orchestration.
- **Inpainting & Mask Layer**:
  - `app/core/inpaint/opencv_engine.py`, `lama_engine.py`, `desktop/core/inpaint_engine.py`: Mask generation, uniform background flat-fill, Telea/LaMa inpainting without alpha smudge bleeding.
- **Typography & Layout Layer**:
  - `app/core/typography/engine.py`: Multiline text layout, contrast-aware font and stroke color selection based on `block.bg_color`, rotated bounding box affine rendering.
- **E2E Testing & Evaluation Layer**:
  - `tests/unit/`: 263 baseline unit tests.
  - `tests/e2e/test_pixiv_10pages.py`: Automated 5-dimension test harness on `D:\baidu\download\baidu\pixiv\test`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | CTD Spatial & Boundary Line Splitting | Split text lines across vertical gaps ($y\_gap > 1.2 \times avg\_h$) or dark boundary barriers in `ctd_engine.py` | M1 | Survey (Parent, E1) |
| F2 | Bubble Connected Component Refinement | Reduce closing kernel (7x7) and prevent whole-page merging in `compute_bubble_labels` | M1 | Survey (E1, E3) |
| F3 | Ray-Casting Dark Boundary Barrier | Direct pixel luminance ray-casting in `can_merge_pair` to reject cross-border merges | M1 | Survey (E1) |
| F4 | Slanted Handwriting Boundary Protection | Rotated $(u,v)$ frame, vector collinearity fallback for short words, prevent AABB border contamination | M1 | Survey (E1, E2) |
| F5 | Fast English Pre-Detection & Hard Gating | Detect English pages and hard-disable Manga-OCR to eliminate Japanese kana hallucinations | M2 | Survey (E2) |
| F6 | EasyOCR 2D Spatial Fragment Sorting | Sort EasyOCR bounding boxes top-to-bottom and left-to-right before joining text lines | M2 | Survey (Parent, E2) |
| F7 | Slanted Upright Crop Normalization | Affine rotate tilted text crops upright before feeding to EasyOCR | M2 | Survey (E1, E2) |
| F8 | Translation Context & Punctuation Guard | Filter out stray colons/fragments (e.g. "：衣服", "：天") and inject manga slang glossary | M2 | Survey (E2) |
| F9 | Aspect-Ratio Onomatopoeia De-biasing | Allow wide rectangular narration banners (aspect > 4.0) to be classified as bubbles/banners | M3 | Survey (E3) |
| F10 | Inverted Dark Box Instant Flat-Fill | Use solid median fill without Gaussian alpha feathering for uniform black/dark boxes | M3 | Survey (E3) |
| F11 | Background-Color-Aware Typography | Respect stored `block.bg_color` to select white text with contrasting stroke on dark boxes | M3 | Survey (E3) |
| F12 | Automated 10-Image E2E Evaluation Suite | Implement complete test script testing all 10 images in `D:\baidu\download\baidu\pixiv\test` | M4 (T1) | Survey (E3) |
| F13 | Zero-Regression Unit Test Guarantee | Maintain 100% pass rate across all 263 unit tests in `pytest tests/unit` | M4 | Survey (E3) |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Bubble Boundary Barrier & Isolation | `ctd_engine.py`, `ocr/base.py` (F1, F2, F3, F4) | none | IN_PROGRESS (Worker: dba9eb96) |
| M2 | OCR Smart Routing & EasyOCR Sorting | `ocr_engine.py`, `easyocr_engine.py`, `pipeline.py` (F5, F6, F7, F8) | none | IN_PROGRESS (Worker: 09d22ff6) |
| M3 | Dark Background Inpainting & Typography | `inpaint_engine.py`, `opencv_engine.py`, `typography/engine.py` (F9, F10, F11) | none | IN_PROGRESS (Worker: 7c870959) |
| M4 | E2E Testing Track & Final Verification | `tests/e2e/test_pixiv_10pages.py`, 10-image harness (F12, F13) | M1, M2, M3 | IN_PROGRESS (Test Writer: 86daeab9) |

## Interface Contracts
### CTD & OCR Base Contract
- `ctd_engine.py`:
  - `_cluster_lines_by_angle(lines, img=None)`: returns `List[List[dict]]` where lines separated by $y\_gap > \max(20, 1.2 \times avg\_h)$ or dark boundary pixels ($I < 90$) are partitioned into separate line groups.
- `app/core/ocr/base.py`:
  - `can_merge_pair(box1, box2, ...)`: returns `False` if ray connecting centroids intersects dark line barrier or if angle divergence cannot be bridged by vector collinearity.
  - `compute_bubble_labels(img)`: uses kernel $\le 7\times 7$ so adjacent bubbles retain distinct positive component labels.

### OCR & Language Perception Contract
- `desktop/core/ocr_engine.py`:
  - `detect_page_language(img, blocks)`: returns `"en"` for English manga, disabling Manga-OCR fallback completely.
  - `_read_block_easyocr(crop)`: sorts detections by $(ymin, xmin)$ before `" ".join()`.

### Inpainting & Typography Contract
- `desktop/core/ocr_engine.py`:
  - `classify_block_type(box, aspect, uniform)`: keeps uniform wide boxes as `"bubble"` / `"narration"`.
- `app/core/inpaint/`:
  - Uniform dark boxes (`block.bg_color == "#000000"` or $lum < 40$) are flat-filled with background median color; skip Gaussian feather blending that bleeds white text edges.
- `app/core/typography/engine.py`:
  - If `block.bg_color` is dark ($lum < 60$), force `text_color_hex = "#FFFFFF"` and stroke `#000000`.

## Code Layout
- `app/core/ocr/ctd_engine.py` — Comic Text Detector clustering & line extraction
- `app/core/ocr/base.py` — Connected component labeling and pair merging
- `desktop/core/ocr_engine.py` — Desktop OCR engine, language detection, EasyOCR sorting
- `app/core/ocr/easyocr_engine.py` — Core EasyOCR engine
- `app/core/inpaint/opencv_engine.py` & `desktop/core/inpaint_engine.py` — Inpainting solid flat-fill
- `app/core/typography/engine.py` — Typography color & rendering
- `tests/unit/` — Existing 263 unit tests
- `tests/e2e/test_pixiv_10pages.py` — Automated 10-image benchmark harness
