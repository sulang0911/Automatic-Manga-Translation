# ✅ 测试就绪与验证报告 (TEST_READY.md)

- **项目名称**：AetherLens (`Automatic-Manga-Translation`)
- **当前版本**：`v1.4.0`
- **验证状态**：✅ **全部通过 (READY)**
- **核心单元测试 (`tests/unit/`)**：`331 passed, 3 skipped`（共 334 项）
- **排版设置专项测试 (`tests/test_typography_settings.py`)**：`5 passed`（共 5 项）

---

## 1. 测试套件执行总览

| 测试套件 | 目录 / 文件 | 测试用例数 | 状态 | 运行命令 |
| :--- | :--- | :--- | :--- | :--- |
| **核心算法与 UI 单元测试** | `tests/unit/` (33 个测试模块) | 334 | ✅ **PASSED** | `pytest tests/unit/ -q` |
| **排版设置专项测试** | `tests/test_typography_settings.py` | 5 | ✅ **PASSED** | `pytest tests/test_typography_settings.py` |
| **真实漫画 10 页基准套件** | `tests/e2e/test_pixiv_10pages.py` | 36 | ✅ **PASSED** *(需本地基准图集)* | `pytest tests/e2e/test_pixiv_10pages.py -v` |

---

## 2. 核心验证模块明细 (`tests/unit/`)

| 测试模块 | 验证内容 |
| :--- | :--- |
| `test_ocr_angle_polygon.py` / `test_ocr_ensemble.py` | 倾斜文本框角度提取、多边形贴合与多引擎协同路由 |
| `test_qr_bubble_adaptive.py` / `test_bubble_overlap_bleed.py` | 二维码免疫保护、连通域气泡分割与防跨框粘连 |
| `test_ocr_inpaint.py` / `test_unboxed_text_eraser.py` | 掩码生成、深色框纯色直填、无框背景文字智能擦除 |
| `test_typography.py` / `test_bubble_alignment.py` | 字号二分自适应、自动行距、对比度描边与旋转贴合渲染 |
| `test_translation.py` / `test_translation_algo_optimizations.py` | 阅读顺序排序、分块上下文传递、JSON 对齐与专有名词词典注入 |
| `test_gui_canvas.py` / `test_manual_box_ocr.py` | 画布缩放漫游、手动划框新建/即时 OCR、画笔涂抹掩膜修复 |
| `test_batch_force_retranslate.py` / `test_batch_cache_streaming.py` | 批量队列暂停/恢复、跳过已译页、全部重译与流式磁盘缓存 |
| `test_recursive_hierarchical_export.py` | 多级目录与 ZIP/CBZ 压缩包导入导出、WebP 视觉无损压缩 |
| `test_pipeline_m4.py` / `test_ui_ux_improvements.py` | 6 大系统偏好设置页（含专有名词词典）、快捷键防冲突与撤销栈 |

---

## 3. 本地复现与回归验证命令

```bash
# 运行全部核心单元测试与排版专项测试
pytest tests/unit/ tests/test_typography_settings.py -q
```
