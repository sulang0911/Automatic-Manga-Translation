"""
app/core/hardware.py
GPU capability detection, Pascal/Maxwell architecture hazard detection, VRAM monitoring,
and aggressive system RAM / CUDA working-set reclamation for 8GB RAM + GTX 1080 Ti systems.
"""
from typing import Dict, Any, Tuple
import os
import sys
import gc
import ctypes
import logging

logger = logging.getLogger(__name__)

# Prevent CUDA allocator fragmentation and OpenCV OpenCL duplicate context on Pascal GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")

try:
    import cv2
    if hasattr(cv2, "ocl") and hasattr(cv2.ocl, "setUseOpenCL"):
        cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

try:
    import torch
    HAS_TORCH = True
    try:
        if hasattr(torch.backends, "cudnn"):
            # Disable cuDNN auto-tuner to prevent multi-GB workspace spikes on varying crop sizes
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
except ImportError:
    torch = None
    HAS_TORCH = False

_VRAM_FRACTION_CONFIGURED = False


def configure_low_resource_runtime(max_vram_fraction: float = 0.72) -> None:
    """
    Configures PyTorch CUDA allocator and cuDNN for 8GB RAM + GTX 1080 Ti (11GB VRAM) stability.
    Prevents PyTorch from over-allocating 100% physical VRAM and spilling into Windows WDDM
    Shared System Memory (which exhausts 8GB RAM and causes system hangs / TDR driver resets).
    """
    global _VRAM_FRACTION_CONFIGURED
    if _VRAM_FRACTION_CONFIGURED:
        return
    _VRAM_FRACTION_CONFIGURED = True

    try:
        if HAS_TORCH:
            try:
                torch.set_num_threads(1)
                if hasattr(torch, "set_num_interop_threads"):
                    torch.set_num_interop_threads(1)
            except Exception:
                pass
            if torch.cuda.is_available():
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.benchmark = False
                if hasattr(torch.cuda, "set_per_process_memory_fraction"):
                    torch.cuda.set_per_process_memory_fraction(max_vram_fraction, 0)
    except Exception:
        pass


def lower_worker_priority() -> None:
    """
    Lowers the OS scheduling priority of the current background worker thread so that
    heavy OCR/Inpaint compute never starves the Windows Desktop Window Manager (dwm.exe),
    mouse input, or the main PyQt6 UI thread.
    """
    try:
        from PyQt6.QtCore import QThread
        curr_thread = QThread.currentThread()
        if curr_thread is not None:
            curr_thread.setPriority(QThread.Priority.LowPriority)
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            thread_handle = kernel32.GetCurrentThread()
            # THREAD_PRIORITY_BELOW_NORMAL = -1
            kernel32.SetThreadPriority(thread_handle, -1)
        except Exception:
            pass


def trim_process_working_set() -> None:
    """
    Forces the OS C/C++ heap and virtual memory manager to reclaim freed NumPy/OpenCV/PyTorch
    memory pages immediately rather than keeping them committed in the process working set.
    Critical for preventing pagefile thrashing and OOM freezes on 8GB RAM Windows systems.
    """
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            try:
                ctypes.windll.psapi.EmptyWorkingSet(handle)
            except Exception:
                pass
            kernel32.SetProcessWorkingSetSize(handle, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass


def is_legacy_pascal_or_maxwell_gpu() -> bool:
    """
    Detects NVIDIA Maxwell (compute capability 5.x) or Pascal (compute capability 6.x) GPUs.
    These architectures (e.g. GTX 1080 Ti) have precision or compatibility quirks with newer CUDA 12 runtimes.
    """
    if not HAS_TORCH or not torch.cuda.is_available():
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
    if not HAS_TORCH or not torch.cuda.is_available():
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


def cleanup_gpu_memory(trim_ram: bool = True) -> None:
    """
    Explicitly triggers Python garbage collection, releases cached PyTorch/Paddle CUDA tensors,
    and trims the OS process working set so 8GB RAM systems never accumulate dead heap pages.
    """
    gc.collect()
    try:
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        if "paddle" in sys.modules:
            import paddle
            if hasattr(paddle, "device") and hasattr(paddle.device, "cuda"):
                paddle.device.cuda.empty_cache()
    except Exception:
        pass
    if trim_ram:
        trim_process_working_set()


_DETACHED_QTHREADS: list = []


def _disconnect_signal_safely(sig) -> None:
    """Disconnect all receivers from a pyqtSignal without raising if empty/deleted."""
    try:
        sig.disconnect()
    except (TypeError, RuntimeError):
        pass


def _release_detached_qthread(worker) -> None:
    """Release a finished detached QThread reference on the GUI thread."""
    try:
        if worker.isRunning():
            return
    except RuntimeError:
        pass
    try:
        _DETACHED_QTHREADS.remove(worker)
    except ValueError:
        pass
    try:
        worker.deleteLater()
    except RuntimeError:
        pass


def retire_qthread(worker, wait_ms: int = 250, signals: tuple = ()) -> None:
    """
    Safely stop or detach a QThread before its parent/reference is dropped.

    Prevents Qt's fatal `QThread: Destroyed while thread '' is still running`
    abort (-1073740791) when a dialog/window closes or a worker slot is
    replaced while the worker thread is still blocked in HTTP or C++ code.
    """
    if worker is None:
        return

    for sig in signals:
        _disconnect_signal_safely(sig)

    try:
        if hasattr(worker, "cancel"):
            worker.cancel()
    except Exception:
        pass

    try:
        running = worker.isRunning()
    except RuntimeError:
        return

    if not running:
        try:
            if worker.parent() is not None:
                worker.setParent(None)
            worker.deleteLater()
        except RuntimeError:
            pass
        return

    try:
        worker.quit()
    except RuntimeError:
        return

    try:
        finished = worker.wait(max(0, int(wait_ms)))
    except RuntimeError:
        return

    if finished:
        try:
            if worker.parent() is not None:
                worker.setParent(None)
            worker.deleteLater()
        except RuntimeError:
            pass
        return

    # Still running after wait_ms: un-parent and pin until `finished` emits so
    # QThread::~QThread() is never called while the OS thread is still alive.
    try:
        from PyQt6.QtCore import Qt

        worker.setParent(None)
        if worker not in _DETACHED_QTHREADS:
            _DETACHED_QTHREADS.append(worker)
        worker.finished.connect(
            lambda w=worker: _release_detached_qthread(w),
            Qt.ConnectionType.QueuedConnection,
        )
        if not worker.isRunning():
            _release_detached_qthread(worker)
    except RuntimeError:
        pass
