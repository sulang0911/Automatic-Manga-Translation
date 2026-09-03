#!/usr/bin/env python3
import torch  # CRITICAL WINDOWS DLL PROTECTION: Must import torch before paddle
import sys
import os

if __name__ == "__main__":
    if "--legacy" in sys.argv:
        from desktop.main import main
    else:
        from app.main import main
    main()

