import os
import sys
import urllib.request
import subprocess
import shutil
import zipfile

ZIP_URL = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "python_embed.zip")
TARGET_DIR = os.path.join(BASE_DIR, "python_env")
MARKER_FILE = os.path.join(TARGET_DIR, ".setup_complete")

def log(msg):
    print(f"[*] {msg}", flush=True)

def extract_zip(zip_path, target_dir):
    log(f"Extracting {zip_path} to {target_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    log("Extraction complete.")

def configure_pth(target_dir):
    pth_file = os.path.join(target_dir, "python310._pth")
    if os.path.exists(pth_file):
        log("Configuring python310._pth to enable site-packages...")
        with open(pth_file, "r") as f:
            content = f.read()
        # Uncomment import site if it's commented out
        if "#import site" in content:
            content = content.replace("#import site", "import site")
            with open(pth_file, "w") as f:
                f.write(content)
            log("python310._pth configured successfully.")
        elif "import site" not in content:
            content += "\nimport site\n"
            with open(pth_file, "w") as f:
                f.write(content)
            log("python310._pth configured by appending 'import site'.")
        else:
            log("python310._pth already configured.")
    else:
        log("Warning: python310._pth not found.")

def main():
    # Detect if we are already running under the local target python
    current_python = os.path.realpath(sys.executable)
    target_python = os.path.realpath(os.path.join(TARGET_DIR, "python.exe"))
    is_running_local = (current_python == target_python)

    if not is_running_local:
        log("Running under external/global Python. Bootstrapping local environment...")
        
        # Remove completion marker if setting up again
        if os.path.exists(MARKER_FILE):
            os.remove(MARKER_FILE)

        # 1. Download Python 3.10.11 Embeddable Zip
        if not os.path.exists(TARGET_DIR):
            os.makedirs(TARGET_DIR)
            
        if not os.path.exists(target_python):
            log("Downloading Python 3.10.11 embeddable package...")
            try:
                urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
                log("Python embeddable package downloaded successfully.")
            except Exception as e:
                log(f"Failed to download python embeddable package: {e}")
                return

            # 2. Extract Zip
            try:
                extract_zip(ZIP_PATH, TARGET_DIR)
            except Exception as e:
                log(f"Failed to extract zip: {e}")
                return
            finally:
                if os.path.exists(ZIP_PATH):
                    os.remove(ZIP_PATH)

            # 3. Configure ._pth file
            configure_pth(TARGET_DIR)
        else:
            log("Local python.exe already exists, skipping download and extraction.")

        # 4. Transfer control to the local python
        log("Transferring execution to local Python...")
        try:
            # We execute setup_venv.py using the newly unzipped local python
            subprocess.run([target_python, __file__], check=True)
            log("Local setup completed successfully.")
        except Exception as e:
            log(f"Error during local python execution: {e}")
            sys.exit(1)
        return

    # --- Below this point, we are running under python_env\python.exe ---
    log("Running setup inside the local Python environment...")

    # Ensure python310._pth is configured (especially if bootstrap happened externally)
    configure_pth(TARGET_DIR)

    # 5. Install pip if not present
    pip_installed = False
    try:
        import pip
        pip_installed = True
        log("pip is already installed.")
    except ImportError:
        pass

    if not pip_installed:
        get_pip_path = os.path.join(BASE_DIR, "get-pip.py")
        log("Downloading get-pip.py...")
        try:
            urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip_path)
            log("get-pip.py downloaded.")
        except Exception as e:
            log(f"Failed to download get-pip.py: {e}")
            return

        log("Installing pip...")
        try:
            # Run get-pip.py to install pip and setuptools locally
            subprocess.run([sys.executable, get_pip_path, "--no-warn-script-location"], check=True)
            log("pip installed successfully.")
        except Exception as e:
            log(f"Failed to run get-pip.py: {e}")
            return
        finally:
            if os.path.exists(get_pip_path):
                os.remove(get_pip_path)

    # 6. Install core dependencies
    log("Installing dependencies (this may take a few minutes)...")
    packages = ["paddleocr", "flask", "flask-cors", "opencv-python", "numpy", "simple-lama-inpainting", "easyocr"]
    try:
        subprocess.run([sys.executable, "-m", "pip", "install"] + packages, check=True)
        log("Core dependencies installed.")
    except Exception as e:
        log(f"Failed to install requirements: {e}")
        return

    # 7. Uninstall CPU paddlepaddle (to avoid conflict with GPU version)
    log("Uninstalling CPU paddlepaddle (to avoid conflict)...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "paddlepaddle"], check=True)
        log("CPU paddlepaddle uninstalled.")
    except Exception as e:
        log(f"No CPU paddlepaddle to uninstall or failed: {e}")

    # 8. Install GPU PaddlePaddle 3.3.1
    log("Installing paddlepaddle-gpu...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "paddlepaddle-gpu==3.3.1", 
            "-i", "https://www.paddlepaddle.org.cn/packages/stable/cu118/"
        ], check=True)
        log("paddlepaddle-gpu installed successfully.")
    except Exception as e:
        log(f"Failed to install paddlepaddle-gpu: {e}")
        return

    # 9. Install CPU versions of torch/torchvision for EasyOCR & LaMa to avoid GPU conflict
    log("Installing CPU version of PyTorch to avoid conflict with paddlepaddle-gpu...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", 
            "--index-url", "https://download.pytorch.org/whl/cpu", 
            "--force-reinstall"
        ], check=True)
        log("CPU PyTorch installed successfully.")
    except Exception as e:
        log(f"Warning: CPU PyTorch installation failed or skipped: {e}")

    # 10. Create completion marker
    try:
        with open(MARKER_FILE, "w") as f:
            f.write("complete")
        log("Setup completion marker created.")
    except Exception as e:
        log(f"Warning: could not write marker file: {e}")

    log("="*50)
    log("ISOLATED PORTABLE PYTHON 3.10 & PADDLEOCR-GPU ENVIRONMENT READY!")
    log(f"Location: {TARGET_DIR}")
    log("="*50)

if __name__ == "__main__":
    main()
