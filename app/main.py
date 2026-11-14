"""
app/main.py
Native Desktop Entry Point for Automatic Manga Translation System.
Launches Apple HIG-conforming PyQt6 desktop workbench.
"""
import sys
import os

# Ensure project root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Optimize thread concurrency and memory safety on Windows for 8GB RAM + GTX 1080 Ti systems
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.5")
os.environ.setdefault("FLAGS_eager_delete_tensor_gb", "0.0")
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")

# CRITICAL WINDOWS DLL PROTECTION: Initialize torch/OpenMP on the main thread
# BEFORE cv2, PyQt6, or background QThread workers load native DLLs.
try:
    import torch
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
except Exception:
    pass

try:
    import cv2
    cv2.setNumThreads(1)
    if hasattr(cv2, "ocl") and hasattr(cv2.ocl, "setUseOpenCL"):
        cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

try:
    from app.core.hardware import configure_low_resource_runtime
    configure_low_resource_runtime()
except Exception:
    pass

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QThreadPool, QTimer
from app.ui.main_window import MainWindow


def main():
    # Enforce single background thread in Qt thread pool to prevent RAM exhaustion
    QThreadPool.globalInstance().setMaxThreadCount(1)

    if hasattr(Qt.ApplicationAttribute, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("AetherLens Manga Translator (Apple HIG)")
    app.setOrganizationName("AetherLens")

    window = MainWindow()
    window.show()
    # Defer session restoration until the window paints its first frame
    QTimer.singleShot(0, window.restore_last_session)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
