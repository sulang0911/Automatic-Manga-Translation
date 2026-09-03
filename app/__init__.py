"""
Automatic Manga / Webtoon Translation Application Package.
"""
import os

# Prevent Windows OpenMP runtime collision between PyTorch and PaddlePaddle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    # Crucial on Windows: PyTorch must be imported before Paddle to avoid shm.dll WinError 127
    import torch
except ImportError:
    pass

__version__ = "1.0.0"
