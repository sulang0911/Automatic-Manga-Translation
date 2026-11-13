"""
Automatic Manga / Webtoon Translation Application Package.
"""
import os

# Prevent Windows OpenMP runtime collision between PyTorch and PaddlePaddle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__version__ = "1.3.0"
