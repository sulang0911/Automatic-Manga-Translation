#!/usr/bin/env python3
import os
import sys

# Enforce single-thread execution and low-memory safety across all BLAS/OpenMP/PyTorch/CUDA backends
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
# BEFORE cv2, paddle, PyQt6, or background QThread workers load native DLLs.
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

if __name__ == "__main__":
    if "--legacy" in sys.argv:
        from desktop.main import main
    else:
        from app.main import main
    main()
