#!/usr/bin/env python3
import os
import sys

# Enforce single-thread execution and memory safety across all BLAS/OpenMP/PyTorch backends
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

# CRITICAL WINDOWS DLL PROTECTION: Must import torch before paddle
try:
    import torch
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
except Exception:
    pass

try:
    import cv2
    cv2.setNumThreads(1)
except Exception:
    pass

if __name__ == "__main__":
    if "--legacy" in sys.argv:
        from desktop.main import main
    else:
        from app.main import main
    main()
