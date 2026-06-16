import os
import sys
import urllib.request
import subprocess
import shutil

INSTALLER_URL = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTALLER_PATH = os.path.join(BASE_DIR, "python_installer.exe")
TARGET_DIR = os.path.join(BASE_DIR, "python_env")
VENV_DIR = os.path.join(BASE_DIR, "venv")

def log(msg):
    print(f"[*] {msg}", flush=True)

def main():
    # 1. Download Python 3.10.11 Installer
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        
    log("Downloading Python 3.10.11 installer...")
    try:
        urllib.request.urlretrieve(INSTALLER_URL, INSTALLER_PATH)
        log("Python installer downloaded successfully.")
    except Exception as e:
        log(f"Failed to download installer: {e}")
        return

    # 2. Run Python Installer Silently
    log(f"Installing Python 3.10.11 to {TARGET_DIR} silently...")
    cmd = [
        INSTALLER_PATH,
        "/quiet",
        "InstallAllUsers=0",
        f"TargetDir={TARGET_DIR}",
        "PrependPath=0",
        "Include_doc=0",
        "Include_test=0"
    ]
    try:
        subprocess.run(cmd, check=True)
        log("Python 3.10.11 installation finished.")
    except Exception as e:
        log(f"Python installation failed: {e}")
        return

    # 3. Verify installed python
    python_exe = os.path.join(TARGET_DIR, "python.exe")
    if not os.path.exists(python_exe):
        log(f"Error: python.exe not found at {python_exe}")
        return
    log(f"Python executable verified at: {python_exe}")

    # 4. Create virtual environment using the installed Python 3.10
    log(f"Creating virtual environment at {VENV_DIR}...")
    if os.path.exists(VENV_DIR):
        log("Removing existing venv...")
        shutil.rmtree(VENV_DIR)
        
    try:
        subprocess.run([python_exe, "-m", "venv", VENV_DIR], check=True)
        log("Virtual environment created successfully.")
    except Exception as e:
        log(f"Failed to create venv: {e}")
        return

    # 5. Clean up installer
    if os.path.exists(INSTALLER_PATH):
        os.remove(INSTALLER_PATH)
        log("Cleaned up installer executable.")

    # 6. Verify venv python
    venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        log(f"Error: venv python not found at {venv_python}")
        return
    log(f"Venv Python verified at: {venv_python}")

    # 7. Upgrade pip
    log("Upgrading pip in virtual environment...")
    try:
        subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        log("pip upgraded.")
    except Exception as e:
        log(f"Warning: failed to upgrade pip: {e}")

    # 8. Install paddleocr and other dependencies
    log("Installing paddleocr and other dependencies...")
    packages = ["paddleocr", "flask", "flask-cors", "opencv-python", "numpy", "simple-lama-inpainting", "easyocr"]
    try:
        subprocess.run([venv_python, "-m", "pip", "install"] + packages, check=True)
        log("Dependencies installed.")
    except Exception as e:
        log(f"Failed to install requirements: {e}")
        return

    # 9. Uninstall CPU paddlepaddle
    log("Uninstalling CPU paddlepaddle (to avoid conflict)...")
    try:
        subprocess.run([venv_python, "-m", "pip", "uninstall", "-y", "paddlepaddle"], check=True)
        log("CPU paddlepaddle uninstalled.")
    except Exception as e:
        log(f"No CPU paddlepaddle to uninstall or failed: {e}")

    # 10. Install GPU PaddlePaddle
    log("Installing paddlepaddle-gpu...")
    try:
        subprocess.run([
            venv_python, "-m", "pip", "install", 
            "paddlepaddle-gpu==2.6.2", 
            "-i", "https://www.paddlepaddle.org.cn/packages/stable/cu118/"
        ], check=True)
        log("paddlepaddle-gpu installed successfully.")
    except Exception as e:
        log(f"Failed to install paddlepaddle-gpu: {e}")
        return


    # 11. Install GPU versions of torch/torchvision for GPU-accelerated EasyOCR & LaMa
    log("Installing GPU version of PyTorch (CUDA 12.1) for EasyOCR / LaMa GPU support...")
    try:
        subprocess.run([
            venv_python, "-m", "pip", "install", 
            "torch", "torchvision", 
            "--index-url", "https://download.pytorch.org/whl/cu121", 
            "--force-reinstall"
        ], check=True)
        log("GPU PyTorch installed successfully.")
    except Exception as e:
        log(f"Warning: GPU PyTorch installation failed or skipped: {e}")

    log("="*50)
    log("ISOLATED PYTHON 3.10 & PADDLEOCR-GPU ENVIRONMENT READY!")
    log(f"Venv location: {VENV_DIR}")
    log("="*50)

if __name__ == "__main__":
    main()
