"""
app/core/pipeline package
Lazy-loads worker classes so lightweight imports (e.g. exporter, utils) do not
pull PyTorch or OCR backends into memory until needed.
"""
import importlib
from typing import Dict, Tuple

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "PipelineWorker": ("app.core.pipeline.pipeline_worker", "PipelineWorker"),
    "BatchWorker": ("app.core.pipeline.batch_worker", "BatchWorker"),
    "MangaExporter": ("app.core.pipeline.exporter", "MangaExporter"),
    "BlockOcrTranslateWorker": ("app.core.pipeline.block_worker", "BlockOcrTranslateWorker"),
    "normalize_domain_slang": ("app.core.pipeline.utils", "normalize_domain_slang"),
    "clean_ocr_syntax": ("app.core.pipeline.utils", "clean_ocr_syntax"),
    "clean_translation_syntax": ("app.core.pipeline.utils", "clean_translation_syntax"),
    "prioritize_english_routing": ("app.core.pipeline.utils", "prioritize_english_routing"),
    "post_process_translation_blocks": ("app.core.pipeline.utils", "post_process_translation_blocks"),
    "normalize_source_lang": ("app.core.pipeline.utils", "normalize_source_lang"),
    "is_auto_source_lang": ("app.core.pipeline.utils", "is_auto_source_lang"),
    "source_lang_to_ocr_lang": ("app.core.pipeline.utils", "source_lang_to_ocr_lang"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(spec[0])
    val = getattr(mod, spec[1])
    globals()[name] = val
    return val


def __dir__():
    return sorted(set(globals()) | set(__all__))
