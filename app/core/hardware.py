"""
app/core/hardware.py
GPU capability detection, Pascal/Maxwell architecture hazard detection, and VRAM monitoring.
"""
from typing import Dict, Any, Tuple
import gc
import logging
import torch

logger = logging.getLogger(__name__)


def is_legacy_pascal_or_maxwell_gpu() -> bool:
    """
    Detects NVIDIA Maxwell (compute capability 5.x) or Pascal (compute capability 6.x) GPUs.
    These architectures (e.g. GTX 1080 Ti) have precision or compatibility quirks with newer CUDA 12 runtimes.
    """
    if not torch.cuda.is_available():
        return False
    try:
        cap = torch.cuda.get_device_capability(0)
        if cap[0] in [5, 6]:
            return True
    except Exception:
        pass

    try:
        name = torch.cuda.get_device_name(0).lower()
        legacy_tokens = ["1080", "1070", "1060", "1050", "titan xp", "p40", "p100", "980", "970", "960"]
        return any(x in name for x in legacy_tokens)
    except Exception:
        return False


def get_gpu_info() -> Dict[str, Any]:
    """Queries and returns GPU availability, device name, compute capability, and VRAM status."""
    available = torch.cuda.is_available()
    if not available:
        return {
            "cuda_available": False,
            "device_name": "None",
            "compute_capability": (0, 0),
            "is_legacy_gpu": False,
            "vram_total_mb": 0.0,
            "vram_free_mb": 0.0,
        }

    try:
        device_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mb = round(free_bytes / (1024 ** 2), 1)
        total_mb = round(total_bytes / (1024 ** 2), 1)
        is_legacy = cap[0] in [5, 6] or is_legacy_pascal_or_maxwell_gpu()
    except Exception as e:
        logger.warning(f"Error querying GPU metrics: {e}")
        device_name = "Unknown CUDA Device"
        cap = (0, 0)
        free_mb, total_mb = 0.0, 0.0
        is_legacy = False

    return {
        "cuda_available": True,
        "device_name": device_name,
        "compute_capability": cap,
        "is_legacy_gpu": is_legacy,
        "vram_total_mb": total_mb,
        "vram_free_mb": free_mb,
    }


def is_vram_constrained(min_free_mb: float = 1200.0) -> bool:
    """Returns True if available free VRAM is lower than min_free_mb."""
    info = get_gpu_info()
    if not info["cuda_available"]:
        return True
    return info["vram_free_mb"] < min_free_mb


def cleanup_gpu_memory() -> None:
    """Explicitly releases cached PyTorch CUDA tensors and triggers garbage collection."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
