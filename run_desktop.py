#!/usr/bin/env python3
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch  # CRITICAL WINDOWS DLL PROTECTION: Must import torch before paddle
import sys

if __name__ == "__main__":
    if "--legacy" in sys.argv:
        from desktop.main import main
    else:
        from app.main import main
    main()

